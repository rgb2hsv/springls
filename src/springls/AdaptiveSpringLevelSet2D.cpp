#include <springls/AdaptiveSpringLevelSet2D.hpp>

#include <springls/DoubleweaveLevelSet2D.hpp>

#include <springls/PointGrid2D.hpp>
#include <springls/SegmentTree2D.hpp>

namespace springls
{
namespace
{
// Same clamp as the uniform level set: the pull is at most 1.5 cells per step.
constexpr float kMaxPullCells = 1.5f;
// A springl is split while it is more than this many times larger than its leaf's cells.
constexpr float kSplitRatio = 1.5f;
// Contour refinement reach (scaled cells), tuned on the Enright test.
constexpr float kRefineProximity = 4.0f;
// Two neighboring springls merge, keeping their size class, once the flow has compressed them until
// together they span no more than this many times their class (a freshly seeded springl spans
// about one).
constexpr float kCompressedMergeLength = 1.0f;
using Block = AdaptiveGrid2D::Block;
}  // namespace

AdaptiveSpringLevelSet2D::AdaptiveSpringLevelSet2D()
{
    mName = "Adaptive Springls";
    mScaledSpringls = true;
    tuning.refineProximity = kRefineProximity;
}

void AdaptiveSpringLevelSet2D::initLevelSet()
{
    mScaledSpringls = true;
    mBand = NarrowBand2D();  // the uniform band is unused
    mGrid.initialize(mInitialLevelSet, adaptiveTuning.maxDepth, adaptiveTuning.bandWidth);
    // No springls yet: the band follows the initial interface. phi is a distance, and a leaf the
    // interface crosses has a corner within a cell diagonal of it.
    const float margin = bandMargin();
    adaptBand([margin](const Block &block) {
        float minAbs = std::abs(block.tileValue);
        if (!block.isTile()) {
            minAbs = std::numeric_limits<float>::max();
            for (float v : block.phi) minAbs = std::min(minAbs, std::abs(v));
        }
        return minAbs < margin + std::sqrt(2.0f) * block.h;
    });
    extractGridContour();
}

bool AdaptiveSpringLevelSet2D::step()
{
    const bool more = SpringLevelSet2D::step();
    if (mResampleEnabled) mergeCompressedSpringls();
    if (verbose && (mIteration - 1) % kLogInterval == 0) {
        int finest = 0;
        for (int index : mGrid.leaves()) finest = std::max(finest, mGrid.blocks()[index].key.depth);
        std::cerr << mName << ": " << mGrid.leaves().size() << " leaves, finest depth " << finest << ", max |phi| at particles " << maxTrackingError()
                  << " cells" << std::endl;
    }
    return more;
}

float AdaptiveSpringLevelSet2D::maxTrackingError() const
{
    float worst = 0.0f;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (!levelSetIncludes(n)) continue;
        const Vec2 p = mManifold.springlParticles[n];
        worst = std::max(worst, std::abs(mGrid.sample(p)) / mGrid.cellSize(p));
    }
    return worst;
}

void AdaptiveSpringLevelSet2D::adaptBand(const std::function<bool(const Block &)> &near)
{
    auto &blocks = mGrid.blocks();
    // Refine: one depth per pass, until every near leaf is at maxDepth.
    int refined = 0;
    for (int pass = 0; pass < mGrid.maxDepth(); pass++) {
        std::vector<AdaptiveGrid2D::Key> marked;
        for (int leaf : mGrid.leaves()) {
            if (blocks[leaf].key.depth < mGrid.maxDepth() && near(blocks[leaf])) marked.push_back(blocks[leaf].key);
        }
        if (marked.empty()) break;
        for (const AdaptiveGrid2D::Key &key : marked) {
            // Earlier refinements (balance) may already have split this block.
            const Vec2 center = Vec2(key.i + 0.5f, key.j + 0.5f) * (AdaptiveGrid2D::kBlockCells * std::ldexp(1.0f, -key.depth));
            const int leaf = mGrid.findLeaf(center);
            if (blocks[leaf].key.depth == key.depth && mGrid.refine(leaf)) refined++;
        }
    }
    // Coarsen, one depth per call: parents whose (leaf) children are all away from the band.
    std::vector<int> parents;
    {
        std::vector<uint8_t> seen(blocks.size(), 0);
        for (int leaf : mGrid.leaves()) {
            const int parent = blocks[leaf].parent;
            if (parent >= 0 && !seen[parent]) {
                seen[parent] = 1;
                parents.push_back(parent);
            }
        }
    }
    std::sort(parents.begin(), parents.end(), [&](int a, int b) {
        return (blocks[a].key.depth != blocks[b].key.depth) ? blocks[a].key.depth > blocks[b].key.depth : blocks[a].key < blocks[b].key;
    });
    int coarsened = 0;
    for (int parent : parents) {
        if (!mGrid.canCoarsen(parent)) continue;
        bool anyNear = false;
        for (int c : blocks[parent].children) anyNear |= near(blocks[c]);
        if (anyNear) continue;
        mGrid.coarsen(parent);
        coarsened++;
    }
    if (refined + coarsened > 0) mGrid.synchronize();
    if (refined > 0 && mSplitOnRefine) splitOversizedSpringls();
}

void AdaptiveSpringLevelSet2D::adaptToParticles()
{
    const float margin = bandMargin();
    Points2 particles;
    particles.reserve(mManifold.springlParticles.size());
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (levelSetIncludes(n)) particles.push_back(mManifold.springlParticles[n]);
    }
    PointGrid2D grid;
    grid.build(particles, std::max(2.0f * margin, 1.0f));
    adaptBand([&](const Block &block) {
        const Vec2 lo = block.origin - Vec2::Constant(margin);
        return grid.anyInBox(lo, lo + Vec2::Constant(block.size() + 2.0f * margin));
    });
}

SegmentTree2D AdaptiveSpringLevelSet2D::makeSpringlTree() const
{
    SegmentTree2D springlTree;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (levelSetIncludes(n)) springlTree.append(mManifold.springlVertexes[2 * n], mManifold.springlVertexes[2 * n + 1], static_cast<int>(n));
    }
    springlTree.finish();
    return springlTree;
}

int AdaptiveSpringLevelSet2D::correctFromSpringls(const SegmentTree2D &springlTree)
{
    const Manifold2D &m = mManifold;
    return mGrid.correctSigns([&](const Vec2 &q, float h, float &d) {
        // Within a cell of a springl, and beside it rather than past one of its ends (where the
        // side is ambiguous): signed by the springl's outward normal.
        const auto hit = springlTree.nearest(q, h);
        if (!hit) return false;
        const size_t s = static_cast<size_t>(hit->id);
        constexpr float kEndTolerance = 1E-6f;
        if (distanceSqr(hit->point, m.springlVertexes[2 * s]) < kEndTolerance || distanceSqr(hit->point, m.springlVertexes[2 * s + 1]) < kEndTolerance) {
            return false;
        }
        const float side = m.springlParticleNormals[s].dot(q - hit->point);
        d = (side < 0.0f) ? -std::sqrt(hit->distanceSqr) : std::sqrt(hit->distanceSqr);
        return std::abs(side) > 0.0f;
    });
}

void AdaptiveSpringLevelSet2D::updateGridVelocities(const SegmentTree2D &springlTree, std::vector<AdaptiveGrid2D::NodeVelocities> &velocities)
{
    mGrid.updateActive();
    const float maxDistance = tuning.unsignedMaxDistance;
    const auto &blocks = mGrid.blocks();
    const auto &active = mGrid.activeLeaves();
    velocities.resize(active.size());
    const int A = static_cast<int>(active.size());
    SPRINGLS_OMP(omp parallel for if (A >= kParallelMinBlocks))
    for (int n = 0; n < A; n++) {
        const Block &block = blocks[active[n]];
        for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
            for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) {
                const Vec2 q = block.nodePosition(a, b);
                const auto hit = springlTree.nearest(q, maxDistance);
                Vec2 velocity = Vec2::Zero();
                if (hit && hit->distanceSqr > 1E-18f) {
                    const float U = std::sqrt(hit->distanceSqr);
                    // Pull toward the nearest springl: -min(U, 1.5 cells) * grad(U), in cell units.
                    velocity = -std::min(U / block.h, kMaxPullCells) * (q - hit->point) / U;
                }
                velocities[n][Block::index(a, b)] = velocity;
            }
        }
    }
}

void AdaptiveSpringLevelSet2D::extractGridContour() { mGrid.extractContour(mManifold.vertexLocations, mManifold.vertexIndexes, mContourScales); }

void AdaptiveSpringLevelSet2D::evolveGrid()
{
    std::vector<AdaptiveGrid2D::NodeVelocities> velocities;  // transient: active blocks only
    const SegmentTree2D springlTree = makeSpringlTree();
    updateGridVelocities(springlTree, velocities);
    mGrid.evolve(tuning.signedEvolveIterations, mCurvatureWeight, velocities);
    if (adaptiveTuning.springlCorrection) correctFromSpringls(springlTree);
    extractGridContour();
}

void AdaptiveSpringLevelSet2D::splitSpringl(size_t index)
{
    Manifold2D &m = mManifold;
    const Vec2 v0 = m.springlVertexes[2 * index], p = m.springlParticles[index], v1 = m.springlVertexes[2 * index + 1];
    const float half = 0.5f * springlScale(index);
    // First half replaces the springl in place; the second half is appended with the same history.
    m.springlParticles[index] = 0.5f * (v0 + p);
    m.springlVertexes[2 * index + 1] = p;
    mSpringlScales[index] = half;
    const size_t other = appendSpringl(0.5f * (p + v1), p, v1, half);
    m.velocities[other] = m.velocities[index];
    if (mTrackingEnabled) {
        m.correspondence[other] = m.correspondence[index];
        m.springlParticleTracking[other] = m.springlParticleTracking[index];
    }
}

size_t AdaptiveSpringLevelSet2D::mergeCompressedSpringls()
{
    Manifold2D &m = mManifold;
    const size_t N = m.springlParticles.size();
    if (mNearestNeighbor.size() != 2 * N) updateNearestNeighbors();
    std::vector<uint8_t> keep(N, 1);
    size_t merges = 0;
    for (size_t i = 0; i < N; i++) {
        if (!keep[i]) continue;
        // Partner: the springl whose start vertex is the neighbor of this springl's end vertex.
        const int neighbor = mNearestNeighbor[2 * i + 1];
        if (neighbor < 0 || (neighbor % 2) != 0) continue;
        const size_t k = static_cast<size_t>(neighbor) / 2;
        if (k == i || !keep[k]) continue;
        const Vec2 v0 = m.springlVertexes[2 * i];
        const Vec2 v1 = m.springlVertexes[2 * k + 1];
        const Vec2 chord = v1 - v0;
        const float length2 = chord.squaredNorm();
        const float limit = kCompressedMergeLength * std::min(springlScale(i), springlScale(k));
        if (length2 <= 0.0f || length2 > limit * limit) continue;
        const Vec2 middle = 0.5f * (m.springlParticles[i] + m.springlParticles[k]);
        const float t = std::clamp((middle - v0).dot(chord) / length2, 0.0f, 1.0f);
        m.springlParticles[i] = v0 + t * chord;
        m.springlVertexes[2 * i + 1] = v1;
        if (mSpringlScales.size() == N) mSpringlScales[i] = std::max(springlScale(i), springlScale(k));
        m.velocities[i] = 0.5f * (m.velocities[i] + m.velocities[k]);
        keep[k] = 0;
        merges++;
    }
    if (merges > 0) {
        compactSpringls(keep);
        m.updateNormals();
        mNearestNeighbor.clear();
    }
    return merges;
}

void AdaptiveSpringLevelSet2D::splitOversizedSpringls()
{
    // Appended halves are visited too, so a springl keeps splitting until it fits its leaf.
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        while (springlScale(n) > kSplitRatio * mGrid.cellSize(mManifold.springlParticles[n])) splitSpringl(n);
    }
    mManifold.updateNormals();
}

void AdaptiveSpringLevelSet2D::evolveLevelSetAndContour()
{
    // The band moves with the springls before the level set follows them.
    adaptToParticles();
    evolveGrid();
    refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
    mManifold.updateNormals();
}

void AdaptiveSpringLevelSet2D::solveContourForFill()
{
    extractGridContour();
    refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
}

// ---------------------------------------------------------------------------------------------
// Adaptive doubleweave
// ---------------------------------------------------------------------------------------------

AdaptiveDoubleweaveLevelSet2D::AdaptiveDoubleweaveLevelSet2D() { mName = "Adaptive Doubleweave"; }

size_t AdaptiveDoubleweaveLevelSet2D::countSpringls(Constellation constellation) const
{
    return static_cast<size_t>(std::count(mManifold.springlConstellations.begin(), mManifold.springlConstellations.end(), static_cast<uint8_t>(constellation)));
}

size_t AdaptiveDoubleweaveLevelSet2D::appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale, Constellation c)
{
    const size_t index = SpringLevelSet2D::appendSpringl(particle, v0, v1, scale);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(c));
    return index;
}

void AdaptiveDoubleweaveLevelSet2D::seedSpringls()
{
    const Points2 &verts = mManifold.vertexLocations;
    for (const Curve &indexes : mManifold.vertexIndexes) {
        const ContourCurve curve(indexes);
        curve.forEachEdge([&](uint32_t a, uint32_t b) {
            appendSpringl(0.5f * (verts[a] + verts[b]), verts[a], verts[b], std::min(contourScale(a), contourScale(b)), Constellation::Primal);
        });
        curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t w) {
            Vec2 particle, v0, v1;
            if (MakeChordSpringl(verts[u], verts[v], verts[w], particle, v0, v1)) {
                appendSpringl(particle, v0, v1, std::min({contourScale(u), contourScale(v), contourScale(w)}), Constellation::Dual);
            }
        });
    }
}

void AdaptiveDoubleweaveLevelSet2D::clearSpringls()
{
    AdaptiveSpringLevelSet2D::clearSpringls();
    mManifold.springlConstellations.clear();
}

void AdaptiveDoubleweaveLevelSet2D::compactSpringls(const std::vector<uint8_t> &keep)
{
    AdaptiveSpringLevelSet2D::compactSpringls(keep);
    size_t out = 0;
    for (size_t i = 0; i < keep.size(); i++) {
        if (keep[i]) mManifold.springlConstellations[out++] = mManifold.springlConstellations[i];
    }
    mManifold.springlConstellations.resize(out);
}

void AdaptiveDoubleweaveLevelSet2D::snapshotForTracking()
{
    AdaptiveSpringLevelSet2D::snapshotForTracking();
    mOldConstellations = mManifold.springlConstellations;
}

bool AdaptiveDoubleweaveLevelSet2D::canTrackFrom(size_t newIndex, size_t oldIndex) const
{
    return newIndex < mManifold.springlConstellations.size() && oldIndex < mOldConstellations.size() &&
           mManifold.springlConstellations[newIndex] == mOldConstellations[oldIndex];
}

bool AdaptiveDoubleweaveLevelSet2D::levelSetIncludes(size_t springl) const
{
    return mActiveConstellation < 0 ||
           (springl < mManifold.springlConstellations.size() && mManifold.springlConstellations[springl] == static_cast<uint8_t>(mActiveConstellation));
}

void AdaptiveDoubleweaveLevelSet2D::splitSpringl(size_t index)
{
    // Same split as the base class; the appended half keeps the constellation.
    const Constellation c = static_cast<Constellation>(mManifold.springlConstellations[index]);
    AdaptiveSpringLevelSet2D::splitSpringl(index);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(c));
}

void AdaptiveDoubleweaveLevelSet2D::swapState(size_t c)
{
    ConstellationState &state = mStates[c];
    std::swap(mGrid, state.grid);
    std::swap(mManifold.vertexLocations, state.vertexLocations);
    std::swap(mManifold.vertexIndexes, state.vertexIndexes);
    std::swap(mContourScales, state.contourScales);
}

bool AdaptiveDoubleweaveLevelSet2D::init()
{
    mActiveConstellation = -1;
    if (!AdaptiveSpringLevelSet2D::init()) return false;
    // Each constellation starts from the combined level set, then follows its own springls.
    for (ConstellationState &state : mStates) {
        state.grid = mGrid;
        state.vertexLocations = mManifold.vertexLocations;
        state.vertexIndexes = mManifold.vertexIndexes;
        state.contourScales = mContourScales;
    }
    evolveConstellations();
    return true;
}

void AdaptiveDoubleweaveLevelSet2D::evolveConstellations()
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

void AdaptiveDoubleweaveLevelSet2D::evolveLevelSetAndContour()
{
    AdaptiveSpringLevelSet2D::evolveLevelSetAndContour();  // combined grid: output and springl splitting
    evolveConstellations();
}

int AdaptiveDoubleweaveLevelSet2D::contract()
{
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    int contracted = 0;
    for (size_t c = 0; c < kConstellationCount; c++) {
        swapState(c);
        contracted += markContractions(keep, [this, c](size_t n) { return mManifold.springlConstellations[n] == c; });
        swapState(c);
    }
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int AdaptiveDoubleweaveLevelSet2D::fill()
{
    refreshSpringlGrid();
    const Points2 springlVertexes = mManifold.springlVertexes;
    const std::vector<uint8_t> constellations = mManifold.springlConstellations;
    constexpr float kQueryMargin = 1.0f;
    std::array<std::vector<std::pair<Vec2, float>>, kConstellationCount> added;  // (particle, fill distance)
    const auto isOpenGap = [&](size_t c, const Vec2 &pt, float scale) {
        const float fillDistance = tuning.fillDistance * scale;
        const float query = fillDistance + kQueryMargin * scale;
        const auto gap = mSpringlGrid.closestSegment(springlVertexes, pt, query, [&](size_t n) { return constellations[n] == c; });
        if (gap.value_or(query) <= fillDistance) return false;
        for (const auto &[q, d] : added[c]) {
            if (distance(q, pt) <= std::max(d, fillDistance)) return false;
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
        const auto add = [&](const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale) {
            appendSpringl(particle, v0, v1, scale, constellation);
            added[c].emplace_back(particle, tuning.fillDistance * scale);
            filled++;
        };
        for (const Curve &indexes : mManifold.vertexIndexes) {
            const ContourCurve curve(indexes);
            if (constellation == Constellation::Primal) {
                curve.forEachEdge([&](uint32_t a, uint32_t b) {
                    const Vec2 pt = 0.5f * (verts[a] + verts[b]);
                    const float scale = std::min(contourScale(a), contourScale(b));
                    if (isOpenGap(c, pt, scale)) add(pt, verts[a], verts[b], scale);
                });
            } else {
                curve.forEachCorner([&](uint32_t u, uint32_t v, uint32_t w) {
                    Vec2 particle, v0, v1;
                    const float scale = std::min({contourScale(u), contourScale(v), contourScale(w)});
                    if (MakeChordSpringl(verts[u], verts[v], verts[w], particle, v0, v1) && isOpenGap(c, particle, scale)) add(particle, v0, v1, scale);
                });
            }
        }
        mActiveConstellation = -1;
        swapState(c);
    }
    return filled;
}
}  // namespace springls
