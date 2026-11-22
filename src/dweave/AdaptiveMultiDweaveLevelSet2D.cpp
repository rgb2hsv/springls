#include <dweave/AdaptiveMultiDweaveLevelSet2D.hpp>

#include <dweave/DweaveLevelSet2D.hpp>

#include <algorithm>

namespace springls
{
AdaptiveMultiDweaveLevelSet2D::AdaptiveMultiDweaveLevelSet2D() { mName = "Adaptive Multi-Object DoubleWeave"; }

size_t AdaptiveMultiDweaveLevelSet2D::countSpringls(Constellation constellation) const
{
    return static_cast<size_t>(std::count(mManifold.springlConstellations.begin(), mManifold.springlConstellations.end(), static_cast<uint8_t>(constellation)));
}

size_t AdaptiveMultiDweaveLevelSet2D::appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale, int label, Constellation c)
{
    const size_t index = SpringLevelSet2D::appendSpringl(particle, v0, v1, scale, label);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(c));
    return index;
}

void AdaptiveMultiDweaveLevelSet2D::seedSpringls()
{
    const Points2 &verts = mManifold.vertexLocations;
    for (size_t k = 0; k < mManifold.vertexIndexes.size(); k++) {
        const int label = curveLabel(k);
        const ContourCurve curve(mManifold.vertexIndexes[k]);
        curve.forEachEdge([&](uint32_t a, uint32_t b) {
            appendSpringl(0.5f * (verts[a] + verts[b]), verts[a], verts[b], std::min(contourScale(a), contourScale(b)), label, Constellation::Primal);
        });
        curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t w) {
            Vec2 particle, v0, v1;
            if (MakeChordSpringl(verts[u], verts[v], verts[w], particle, v0, v1)) {
                appendSpringl(particle, v0, v1, std::min({contourScale(u), contourScale(v), contourScale(w)}), label, Constellation::Dual);
            }
        });
    }
}

void AdaptiveMultiDweaveLevelSet2D::clearSpringls()
{
    AdaptiveMultiSpringLevelSet2D::clearSpringls();
    mManifold.springlConstellations.clear();
}

void AdaptiveMultiDweaveLevelSet2D::compactSpringls(const std::vector<uint8_t> &keep)
{
    AdaptiveMultiSpringLevelSet2D::compactSpringls(keep);
    size_t out = 0;
    for (size_t i = 0; i < keep.size(); i++) {
        if (keep[i]) mManifold.springlConstellations[out++] = mManifold.springlConstellations[i];
    }
    mManifold.springlConstellations.resize(out);
}

void AdaptiveMultiDweaveLevelSet2D::snapshotForTracking()
{
    AdaptiveMultiSpringLevelSet2D::snapshotForTracking();
    mOldConstellations = mManifold.springlConstellations;
}

bool AdaptiveMultiDweaveLevelSet2D::canTrackFrom(size_t newIndex, size_t oldIndex) const
{
    return AdaptiveMultiSpringLevelSet2D::canTrackFrom(newIndex, oldIndex) && newIndex < mManifold.springlConstellations.size() &&
           oldIndex < mOldConstellations.size() && mManifold.springlConstellations[newIndex] == mOldConstellations[oldIndex];
}

bool AdaptiveMultiDweaveLevelSet2D::canAttract(size_t springl, size_t other) const
{
    return AdaptiveMultiSpringLevelSet2D::canAttract(springl, other) && mManifold.springlConstellations[springl] == mManifold.springlConstellations[other];
}

bool AdaptiveMultiDweaveLevelSet2D::levelSetIncludes(size_t springl) const
{
    return mActiveConstellation < 0 ||
           (springl < mManifold.springlConstellations.size() && inConstellation(springl, static_cast<size_t>(mActiveConstellation)));
}

void AdaptiveMultiDweaveLevelSet2D::splitSpringl(size_t index)
{
    // Same split as the base class (which keeps the label); the appended half keeps the constellation.
    const uint8_t c = mManifold.springlConstellations[index];
    AdaptiveMultiSpringLevelSet2D::splitSpringl(index);
    mManifold.springlConstellations.push_back(c);
}

void AdaptiveMultiDweaveLevelSet2D::swapState(size_t c)
{
    ConstellationState &state = mStates[c];
    std::swap(mGrid, state.grid);
    std::swap(mManifold.vertexLocations, state.vertexLocations);
    std::swap(mManifold.vertexIndexes, state.vertexIndexes);
    std::swap(mManifold.curveLabels, state.curveLabels);
    std::swap(mContourScales, state.contourScales);
}

bool AdaptiveMultiDweaveLevelSet2D::init()
{
    mActiveConstellation = -1;
    if (!AdaptiveMultiSpringLevelSet2D::init()) return false;
    // Each constellation starts from the combined grid, then follows its own springls.
    for (ConstellationState &state : mStates) {
        state.grid = mGrid;
        state.vertexLocations = mManifold.vertexLocations;
        state.vertexIndexes = mManifold.vertexIndexes;
        state.curveLabels = mManifold.curveLabels;
        state.contourScales = mContourScales;
    }
    evolveConstellations();
    return true;
}

void AdaptiveMultiDweaveLevelSet2D::evolveConstellations()
{
    for (size_t c = 0; c < kConstellationCount; c++) {
        swapState(c);
        mActiveConstellation = static_cast<int>(c);
        mSplitOnRefine = false;  // springls follow the combined grid only
        adaptToParticles();
        evolveGrid();
        refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
        mSplitOnRefine = true;
        mActiveConstellation = -1;
        swapState(c);
    }
}

void AdaptiveMultiDweaveLevelSet2D::evolveLevelSetAndContour()
{
    AdaptiveMultiSpringLevelSet2D::evolveLevelSetAndContour();  // combined grid: output and springl splitting
    evolveConstellations();
}

int AdaptiveMultiDweaveLevelSet2D::contract()
{
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    int contracted = 0;
    for (size_t c = 0; c < kConstellationCount; c++) {
        swapState(c);
        // markContractions measures each springl against its own region's curves.
        contracted += markContractions(keep, [this, c](size_t n) { return inConstellation(n, c); });
        swapState(c);
    }
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int AdaptiveMultiDweaveLevelSet2D::fill()
{
    refreshSpringlGrid();
    const Points2 springlVertexes = mManifold.springlVertexes;
    const std::vector<uint8_t> constellations = mManifold.springlConstellations;
    const std::vector<int> labels = mManifold.springlLabels;
    constexpr float kQueryMargin = 1.0f;
    struct Added {
        Vec2 particle;
        float fillDistance;
        int label;
    };
    std::array<std::vector<Added>, kConstellationCount> added;
    const auto isOpenGap = [&](size_t c, int label, const Vec2 &pt, float scale) {
        const float fillDistance = tuning.fillDistance * scale;
        const float query = fillDistance + kQueryMargin * scale;
        const auto gap = mSpringlGrid.closestSegment(springlVertexes, pt, query, [&](size_t n) { return constellations[n] == c && labels[n] == label; });
        if (gap.value_or(query) <= fillDistance) return false;
        for (const Added &a : added[c]) {
            if (a.label == label && distance(a.particle, pt) <= std::max(a.fillDistance, fillDistance)) return false;
        }
        return true;
    };
    int filled = 0;
    for (size_t c = 0; c < kConstellationCount; c++) {
        const Constellation constellation = static_cast<Constellation>(c);
        swapState(c);
        mActiveConstellation = static_cast<int>(c);
        solveContourForFill();
        const Points2 &verts = mManifold.vertexLocations;
        const auto add = [&](const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale, int label) {
            appendSpringl(particle, v0, v1, scale, label, constellation);
            added[c].push_back({particle, tuning.fillDistance * scale, label});
            filled++;
        };
        for (size_t k = 0; k < mManifold.vertexIndexes.size(); k++) {
            const int label = curveLabel(k);
            const ContourCurve curve(mManifold.vertexIndexes[k]);
            if (constellation == Constellation::Primal) {
                curve.forEachEdge([&](uint32_t a, uint32_t b) {
                    const Vec2 pt = 0.5f * (verts[a] + verts[b]);
                    const float scale = std::min(contourScale(a), contourScale(b));
                    if (isOpenGap(c, label, pt, scale)) add(pt, verts[a], verts[b], scale, label);
                });
            } else {
                curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t w) {
                    Vec2 particle, v0, v1;
                    const float scale = std::min({contourScale(u), contourScale(v), contourScale(w)});
                    if (MakeChordSpringl(verts[u], verts[v], verts[w], particle, v0, v1) && isOpenGap(c, label, particle, scale)) {
                        add(particle, v0, v1, scale, label);
                    }
                });
            }
        }
        mActiveConstellation = -1;
        swapState(c);
    }
    return filled;
}
}  // namespace springls
