#include <springls/DoubleweaveLevelSet2D.hpp>

#include <springls/SegmentTree2D.hpp>

#include <cassert>

namespace springls
{
ContourCurve::ContourCurve(const Curve &curve) : vertices(curve)
{
    if (vertices.size() > 1 && vertices.front() == vertices.back()) {
        vertices.pop_back();
        closed = true;
    }
}

bool MakeChordSpringl(const Vec2 &u, const Vec2 &v, const Vec2 &w, Vec2 &particle, Vec2 &v0, Vec2 &v1)
{
    v0 = 0.5f * (u + v);
    v1 = 0.5f * (v + w);
    const Vec2 chord = v1 - v0;
    const float chordLength = chord.norm();
    if (!(chordLength >= SpringLevelSet2D::kMinEdgeLength)) return false;  // also rejects NaN
    // The chord between edge midpoints cuts inside the contour on convex stretches (and outside
    // on concave ones). Translate it, keeping its direction, so it passes through v; v then
    // becomes the particle and stays in line with both arms.
    const Vec2 tangent = chord / chordLength;
    const float t = std::clamp((v - v0).dot(tangent), 0.0f, chordLength);
    const Vec2 shift = v - (v0 + t * tangent);
    v0 += shift;
    v1 += shift;
    particle = v0 + t * tangent;
    return true;
}

DoubleweaveLevelSet2D::DoubleweaveLevelSet2D() { mName = "Doubleweave"; }

size_t DoubleweaveLevelSet2D::countSpringls(Constellation constellation) const
{
    return static_cast<size_t>(std::count(mManifold.springlConstellations.begin(), mManifold.springlConstellations.end(), static_cast<uint8_t>(constellation)));
}

size_t DoubleweaveLevelSet2D::appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, Constellation constellation)
{
    const size_t index = SpringLevelSet2D::appendSpringl(particle, v0, v1);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(constellation));
    return index;
}

void DoubleweaveLevelSet2D::seedSpringls()
{
    const Points2 &verts = mManifold.vertexLocations;
    for (const Curve &indexes : mManifold.vertexIndexes) {
        const ContourCurve curve(indexes);
        curve.forEachEdge([&](uint32_t a, uint32_t b) { appendSpringl(0.5f * (verts[a] + verts[b]), verts[a], verts[b], Constellation::Primal); });
        curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t w) {
            Vec2 particle, v0, v1;
            if (MakeChordSpringl(verts[u], verts[v], verts[w], particle, v0, v1)) appendSpringl(particle, v0, v1, Constellation::Dual);
        });
    }
}

void DoubleweaveLevelSet2D::clearSpringls()
{
    SpringLevelSet2D::clearSpringls();
    mManifold.springlConstellations.clear();
}

void DoubleweaveLevelSet2D::compactSpringls(const std::vector<uint8_t> &keep)
{
    SpringLevelSet2D::compactSpringls(keep);
    size_t out = 0;
    for (size_t i = 0; i < keep.size(); i++) {
        if (keep[i]) mManifold.springlConstellations[out++] = mManifold.springlConstellations[i];
    }
    mManifold.springlConstellations.resize(out);
}

void DoubleweaveLevelSet2D::snapshotForTracking()
{
    SpringLevelSet2D::snapshotForTracking();
    mOldConstellations = mManifold.springlConstellations;
}

bool DoubleweaveLevelSet2D::canTrackFrom(size_t newIndex, size_t oldIndex) const
{
    return newIndex < mManifold.springlConstellations.size() && oldIndex < mOldConstellations.size() &&
           mManifold.springlConstellations[newIndex] == mOldConstellations[oldIndex];
}

bool DoubleweaveLevelSet2D::refinesContourWith(size_t particle) const
{
    return mActiveConstellation < 0 ||
           (particle < mManifold.springlConstellations.size() && mManifold.springlConstellations[particle] == static_cast<uint8_t>(mActiveConstellation));
}

bool DoubleweaveLevelSet2D::init()
{
    for (ConstellationLevelSet &state : mConstellations) state = ConstellationLevelSet();
    mActiveConstellation = -1;
    if (!SpringLevelSet2D::init()) return false;
    ensureConstellationLevelSets();
    evolveConstellationLevelSets();
    return true;
}

void DoubleweaveLevelSet2D::swapConstellation(size_t w)
{
    ConstellationLevelSet &state = mConstellations[w];
    std::swap(mBand, state.band);
    std::swap(mUnsignedLevelSet, state.unsignedLevelSet);
    std::swap(mManifold.vertexLocations, state.vertexLocations);
    std::swap(mManifold.vertexIndexes, state.vertexIndexes);
}

void DoubleweaveLevelSet2D::ensureConstellationLevelSets()
{
    for (ConstellationLevelSet &state : mConstellations) {
        if (state.valid) continue;
        state.band = mBand;
        state.unsignedLevelSet = mUnsignedLevelSet;
        state.vertexLocations = mManifold.vertexLocations;
        state.vertexIndexes = mManifold.vertexIndexes;
        state.valid = true;
    }
}

void DoubleweaveLevelSet2D::updateConstellationUnsignedLevelSet(size_t w)
{
    const float maxDistance = tuning.unsignedMaxDistance;
    mUnsignedLevelSet.resize(mBand.phi().width(), mBand.phi().height());
    SegmentTree2D tree;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (mManifold.springlConstellations[n] == w) tree.append(mManifold.springlVertexes[2 * n], mManifold.springlVertexes[2 * n + 1], static_cast<int>(n));
    }
    tree.finish();
    const int W = mUnsignedLevelSet.width(), H = mUnsignedLevelSet.height();
    SPRINGLS_OMP(omp parallel for if (W * H >= kParallelMinHeavyItems))
    for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++) mUnsignedLevelSet(i, j) = tree.distance(Vec2(i, j), maxDistance);
    }
}

void DoubleweaveLevelSet2D::evolveConstellationLevelSets()
{
    ensureConstellationLevelSets();
    for (size_t w = 0; w < kConstellationCount; w++) {
        swapConstellation(w);
        updateConstellationUnsignedLevelSet(w);
        for (int i = 0; i < tuning.signedEvolveIterations; i++) updateSignedLevelSet();
        mActiveConstellation = static_cast<int>(w);
        solveContourForFill();
        mActiveConstellation = -1;
        swapConstellation(w);
    }
}

int DoubleweaveLevelSet2D::contract()
{
    if (doubleweaveTuning.contractAgainstCombinedContour) return SpringLevelSet2D::contract();
    ensureConstellationLevelSets();
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    int contracted = 0;
    for (size_t w = 0; w < kConstellationCount; w++) {
        swapConstellation(w);
        contracted += markContractions(keep, [this, w](size_t n) { return mManifold.springlConstellations[n] == w; });
        swapConstellation(w);
    }
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int DoubleweaveLevelSet2D::fill()
{
    ensureConstellationLevelSets();
    // Gap tests per constellation, against the springls that exist before this pass.
    const size_t SN = mManifold.springlParticles.size();
    refreshSpringlGrid();
    const Points2 springlVertexes = mManifold.springlVertexes;
    const std::vector<uint8_t> constellations = mManifold.springlConstellations;
    constexpr float kQueryMargin = 1.0f;
    const float queryDistance = tuning.fillDistance + kQueryMargin;
    // Particles added during this pass, so later candidates do not refill the same site.
    std::array<Points2, kConstellationCount> added;
    const auto isOpenGap = [&](Constellation c, const Vec2 &pt) {
        const uint8_t tag = static_cast<uint8_t>(c);
        const auto gap = mSpringlGrid.closestSegment(springlVertexes, pt, queryDistance, [&](size_t n) { return constellations[n] == tag; });
        if (gap.value_or(queryDistance) <= tuning.fillDistance) return false;
        for (const Vec2 &q : added[tag]) {
            if (distance(q, pt) <= tuning.fillDistance) return false;
        }
        return true;
    };
    int filled = 0;
    const auto add = [&](const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, Constellation c) {
        appendSpringl(particle, v0, v1, c);
        added[static_cast<size_t>(c)].push_back(particle);
        filled++;
    };
    if (doubleweaveTuning.fillFromOtherConstellation && mNearestNeighbor.size() == 2 * SN) {
        // A joint between neighboring springls of one constellation marks a particle site of the other.
        for (size_t e = 0; e < 2 * SN; e++) {
            const int nbr = mNearestNeighbor[e];
            if (nbr < 0) continue;
            const size_t a = e / 2, b = static_cast<size_t>(nbr) / 2;
            if (a == b || (e % 2) == (static_cast<size_t>(nbr) % 2)) continue;  // a chain joint links an end to a start
            const Constellation target = (constellations[a] == static_cast<uint8_t>(Constellation::Primal)) ? Constellation::Dual : Constellation::Primal;
            const bool aIsEarlier = (e % 2) == 1;  // arms run from the earlier particle to the later one
            const Vec2 v0 = mManifold.springlParticles[aIsEarlier ? a : b];
            const Vec2 v1 = mManifold.springlParticles[aIsEarlier ? b : a];
            const Vec2 joint = 0.5f * (mManifold.springlVertexes[e] + mManifold.springlVertexes[nbr]);
            const Vec2 chord = v1 - v0;
            const float chordLength = chord.norm();
            if (!(chordLength >= kMinEdgeLength)) continue;
            const Vec2 tangent = chord / chordLength;
            const Vec2 particle = v0 + std::clamp((joint - v0).dot(tangent), 0.0f, chordLength) * tangent;
            if (isOpenGap(target, particle)) add(particle, v0, v1, target);
        }
    }
    // Each constellation fills from its own iso-contour.
    for (size_t w = 0; w < kConstellationCount; w++) {
        const Constellation c = static_cast<Constellation>(w);
        swapConstellation(w);
        mActiveConstellation = static_cast<int>(w);
        solveContourForFill();
        mActiveConstellation = -1;
        const Points2 &verts = mManifold.vertexLocations;
        for (const Curve &indexes : mManifold.vertexIndexes) {
            const ContourCurve curve(indexes);
            if (c == Constellation::Primal) {
                curve.forEachEdge([&](uint32_t a, uint32_t b) {
                    const Vec2 pt = 0.5f * (verts[a] + verts[b]);
                    if (isOpenGap(c, pt)) add(pt, verts[a], verts[b], c);
                });
            } else {
                curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t x) {
                    Vec2 particle, v0, v1;
                    if (MakeChordSpringl(verts[u], verts[v], verts[x], particle, v0, v1) && isOpenGap(c, particle)) add(particle, v0, v1, c);
                });
            }
        }
        swapConstellation(w);
    }
    return filled;
}
}  // namespace springls
