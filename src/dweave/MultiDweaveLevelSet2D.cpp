#include <dweave/MultiDweaveLevelSet2D.hpp>

#include <dweave/DweaveLevelSet2D.hpp>
#include <spatial/SegmentTree2D.hpp>

#include <algorithm>

namespace springls
{
MultiDweaveLevelSet2D::MultiDweaveLevelSet2D() { mName = "Multi-Object DoubleWeave"; }

size_t MultiDweaveLevelSet2D::countSpringls(Constellation constellation) const
{
    return static_cast<size_t>(std::count(mManifold.springlConstellations.begin(), mManifold.springlConstellations.end(), static_cast<uint8_t>(constellation)));
}

size_t MultiDweaveLevelSet2D::appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, int label, Constellation constellation)
{
    const size_t index = SpringLevelSet2D::appendSpringl(particle, v0, v1, 1.0f, label);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(constellation));
    return index;
}

void MultiDweaveLevelSet2D::seedSpringls()
{
    const Points2 &verts = mManifold.vertexLocations;
    for (size_t c = 0; c < mManifold.vertexIndexes.size(); c++) {
        const int label = curveLabel(c);
        const ContourCurve curve(mManifold.vertexIndexes[c]);
        curve.forEachEdge([&](uint32_t a, uint32_t b) { appendSpringl(0.5f * (verts[a] + verts[b]), verts[a], verts[b], label, Constellation::Primal); });
        curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t w) {
            Vec2 particle, v0, v1;
            if (MakeChordSpringl(verts[u], verts[v], verts[w], particle, v0, v1)) appendSpringl(particle, v0, v1, label, Constellation::Dual);
        });
    }
}

void MultiDweaveLevelSet2D::clearSpringls()
{
    MultiSpringLevelSet2D::clearSpringls();
    mManifold.springlConstellations.clear();
}

void MultiDweaveLevelSet2D::compactSpringls(const std::vector<uint8_t> &keep)
{
    MultiSpringLevelSet2D::compactSpringls(keep);
    size_t out = 0;
    for (size_t i = 0; i < keep.size(); i++) {
        if (keep[i]) mManifold.springlConstellations[out++] = mManifold.springlConstellations[i];
    }
    mManifold.springlConstellations.resize(out);
}

void MultiDweaveLevelSet2D::snapshotForTracking()
{
    MultiSpringLevelSet2D::snapshotForTracking();
    mOldConstellations = mManifold.springlConstellations;
}

bool MultiDweaveLevelSet2D::canTrackFrom(size_t newIndex, size_t oldIndex) const
{
    return MultiSpringLevelSet2D::canTrackFrom(newIndex, oldIndex) && newIndex < mManifold.springlConstellations.size() &&
           oldIndex < mOldConstellations.size() && mManifold.springlConstellations[newIndex] == mOldConstellations[oldIndex];
}

bool MultiDweaveLevelSet2D::canAttract(size_t springl, size_t other) const
{
    return MultiSpringLevelSet2D::canAttract(springl, other) && mManifold.springlConstellations[springl] == mManifold.springlConstellations[other];
}

bool MultiDweaveLevelSet2D::refinesContourWith(size_t particle) const
{
    return mActiveConstellation < 0 ||
           (particle < mManifold.springlConstellations.size() && inConstellation(particle, static_cast<size_t>(mActiveConstellation)));
}

bool MultiDweaveLevelSet2D::init()
{
    for (ConstellationLevelSet &state : mConstellations) state = ConstellationLevelSet();
    mActiveConstellation = -1;
    if (!MultiSpringLevelSet2D::init()) return false;
    ensureConstellationLevelSets();
    evolveConstellationLevelSets();
    return true;
}

void MultiDweaveLevelSet2D::swapConstellation(size_t w)
{
    ConstellationLevelSet &state = mConstellations[w];
    std::swap(mMultiBand, state.band);
    std::swap(mUnsignedLevelSet, state.unsignedLevelSet);
    std::swap(mManifold.vertexLocations, state.vertexLocations);
    std::swap(mManifold.vertexIndexes, state.vertexIndexes);
    std::swap(mManifold.curveLabels, state.curveLabels);
}

void MultiDweaveLevelSet2D::ensureConstellationLevelSets()
{
    for (ConstellationLevelSet &state : mConstellations) {
        if (state.valid) continue;
        state.band = mMultiBand;
        state.unsignedLevelSet = mUnsignedLevelSet;
        state.vertexLocations = mManifold.vertexLocations;
        state.vertexIndexes = mManifold.vertexIndexes;
        state.curveLabels = mManifold.curveLabels;
        state.valid = true;
    }
}

void MultiDweaveLevelSet2D::updateConstellationUnsignedLevelSet(size_t w)
{
    const float maxDistance = tuning.unsignedMaxDistance;
    const Vec2i size = levelSetSize();
    mUnsignedLevelSet.resize(size.x(), size.y());
    SegmentTree2D tree;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (inConstellation(n, w)) tree.append(mManifold.springlVertexes[2 * n], mManifold.springlVertexes[2 * n + 1], static_cast<int>(n));
    }
    tree.finish();
    const int W = mUnsignedLevelSet.width(), H = mUnsignedLevelSet.height();
    SPRINGLS_OMP(omp parallel for if (W * H >= kParallelMinHeavyItems))
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) mUnsignedLevelSet(i, j) = tree.distance(Vec2(i, j), maxDistance);
    }
}

void MultiDweaveLevelSet2D::evolveConstellationLevelSets()
{
    ensureConstellationLevelSets();
    for (size_t w = 0; w < kConstellationCount; w++) {
        swapConstellation(w);
        updateConstellationUnsignedLevelSet(w);
        for (int i = 0; i < tuning.signedEvolveIterations; i++) updateMultiLevelSet();
        if (tuning.springlCorrection) correctFromSpringls([this, w](size_t n) { return inConstellation(n, w); });
        mActiveConstellation = static_cast<int>(w);
        solveContourForFill();
        mActiveConstellation = -1;
        swapConstellation(w);
    }
}

int MultiDweaveLevelSet2D::contract()
{
    ensureConstellationLevelSets();
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    int contracted = 0;
    for (size_t w = 0; w < kConstellationCount; w++) {
        swapConstellation(w);
        // markContractions measures each springl against its own region's curves.
        contracted += markContractions(keep, [this, w](size_t n) { return inConstellation(n, w); });
        swapConstellation(w);
    }
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int MultiDweaveLevelSet2D::fill()
{
    ensureConstellationLevelSets();
    // Gap tests per constellation and region, against the springls that exist before this pass.
    refreshSpringlGrid();
    const Points2 springlVertexes = mManifold.springlVertexes;
    const std::vector<uint8_t> constellations = mManifold.springlConstellations;
    const std::vector<int> labels = mManifold.springlLabels;
    constexpr float kQueryMargin = 1.0f;
    const float queryDistance = tuning.fillDistance + kQueryMargin;
    // Particles added during this pass (with their labels), so later candidates do not refill the same site.
    std::array<std::vector<std::pair<Vec2, int>>, kConstellationCount> added;
    const auto isOpenGap = [&](Constellation c, int label, const Vec2 &pt) {
        const uint8_t tag = static_cast<uint8_t>(c);
        const auto gap =
            mSpringlGrid.closestSegment(springlVertexes, pt, queryDistance, [&](size_t n) { return constellations[n] == tag && labels[n] == label; });
        if (gap.value_or(queryDistance) <= tuning.fillDistance) return false;
        for (const auto &[q, l] : added[tag]) {
            if (l == label && distance(q, pt) <= tuning.fillDistance) return false;
        }
        return true;
    };
    int filled = 0;
    // Each constellation fills from its own labeled curves.
    for (size_t w = 0; w < kConstellationCount; w++) {
        const Constellation c = static_cast<Constellation>(w);
        swapConstellation(w);
        mActiveConstellation = static_cast<int>(w);
        solveContourForFill();
        mActiveConstellation = -1;
        const Points2 &verts = mManifold.vertexLocations;
        const auto add = [&](const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, int label) {
            appendSpringl(particle, v0, v1, label, c);
            added[w].emplace_back(particle, label);
            filled++;
        };
        for (size_t k = 0; k < mManifold.vertexIndexes.size(); k++) {
            const int label = curveLabel(k);
            const ContourCurve curve(mManifold.vertexIndexes[k]);
            if (c == Constellation::Primal) {
                curve.forEachEdge([&](uint32_t a, uint32_t b) {
                    const Vec2 pt = 0.5f * (verts[a] + verts[b]);
                    if (isOpenGap(c, label, pt)) add(pt, verts[a], verts[b], label);
                });
            } else {
                curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t x) {
                    Vec2 particle, v0, v1;
                    if (MakeChordSpringl(verts[u], verts[v], verts[x], particle, v0, v1) && isOpenGap(c, label, particle)) add(particle, v0, v1, label);
                });
            }
        }
        swapConstellation(w);
    }
    return filled;
}
}  // namespace springls
