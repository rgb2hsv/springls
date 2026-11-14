#include <springls/AdaptiveSpringLevelSet3D.hpp>
#include <springls/Profile.hpp>

#include <springls/DoubleweaveLevelSet3D.hpp>
#include <springls/PointGrid3D.hpp>
#include <springls/TriangleTree3D.hpp>

#include <optional>

namespace springls
{

namespace
{
// Same clamp as the uniform level set: the pull is at most 1.5 cells per step.
constexpr float kMaxPullCells = 1.5f;
// A springl is split while it is more than this many times larger than its leaf's cells.
constexpr float kSplitRatio = 1.5f;
// Iso-surface refinement reach (scaled cells), as in AdaptiveSpringLevelSet2D.
constexpr float kRefineProximity = 4.0f;
using Block = AdaptiveGrid3D::Block;
}  // namespace

AdaptiveSpringLevelSet3D::AdaptiveSpringLevelSet3D(ElementType elementType) : SpringLevelSet3D(elementType)
{
    mName = "Adaptive Springls 3D";
    mScaledSpringls = true;
    tuning.refineProximity = kRefineProximity;
}

void AdaptiveSpringLevelSet3D::initLevelSet()
{
    mScaledSpringls = true;
    mBand = NarrowBand3D();  // the uniform band is unused
    mGrid.setActiveBand(adaptiveTuning.activeCells, adaptiveTuning.activeMargin);
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
        return minAbs < margin + std::sqrt(3.0f) * block.h;
    });
    extractGridSurface();
}

bool AdaptiveSpringLevelSet3D::step()
{
    const bool more = SpringLevelSet3D::step();
    if (verbose && (mIteration - 1) % kLogInterval == 0) {
        std::cerr << mName << ": " << mGrid.leaves().size() << " leaves, finest depth " << mGrid.finestDepth() << ", max |phi| at particles "
                  << maxTrackingError() << " cells" << std::endl;
    }
    return more;
}

float AdaptiveSpringLevelSet3D::maxTrackingError() const
{
    float worst = 0.0f;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (!levelSetIncludes(n)) continue;
        const Vec3 &p = mManifold.springlParticles[n];
        worst = std::max(worst, std::abs(mGrid.sample(p)) / mGrid.cellSize(p));
    }
    return worst;
}

Vec3 AdaptiveSpringLevelSet3D::levelSetGradient(const Vec3 &p) const
{
    const float h = mGrid.cellSize(p);
    const auto v = [&](const Vec3 &d) { return mGrid.sample(p + h * d); };
    const Vec3 grad(v(Vec3::UnitX()) - v(-Vec3::UnitX()), v(Vec3::UnitY()) - v(-Vec3::UnitY()), v(Vec3::UnitZ()) - v(-Vec3::UnitZ()));
    return grad / std::max(1E-6f, grad.norm());
}

void AdaptiveSpringLevelSet3D::adaptBand(const std::function<bool(const Block &)> &near)
{
    ProfileScope profileScope("adaptBand");
    auto &blocks = mGrid.blocks();
    std::optional<ProfileScope> phase(std::in_place, "adaptBand: refine");
    // Refine: one depth per pass, until every near leaf is at maxDepth.
    int refined = 0;
    for (int pass = 0; pass < mGrid.maxDepth(); pass++) {
        std::vector<AdaptiveGrid3D::Key> marked;
        for (int leaf : mGrid.leaves()) {
            if (blocks[leaf].key.depth < mGrid.maxDepth() && near(blocks[leaf])) marked.push_back(blocks[leaf].key);
        }
        if (marked.empty()) break;
        for (const AdaptiveGrid3D::Key &key : marked) {
            // Earlier refinements (balance) may already have split this block.
            const Vec3 center = Vec3(key.i + 0.5f, key.j + 0.5f, key.k + 0.5f) * (AdaptiveGrid3D::kBlockCells * std::ldexp(1.0f, -key.depth));
            const int leaf = mGrid.findLeaf(center);
            if (blocks[leaf].key.depth == key.depth && mGrid.refine(leaf)) refined++;
        }
    }
    phase.emplace("adaptBand: coarsen");
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
    // Whether any child is near the band: read-only, so in parallel; the coarsening itself stays
    // sequential (each one changes what keeps the grid balanced for the next).
    const int P = static_cast<int>(parents.size());
    std::vector<uint8_t> anyNear(P, 0);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 16) if (P >= kParallelMinBlocks))
    for (int k = 0; k < P; k++) {
        for (int c : blocks[parents[k]].children) {
            if (near(blocks[c])) {
                anyNear[k] = 1;
                break;
            }
        }
    }
    int coarsened = 0;
    for (int k = 0; k < P; k++) {
        if (anyNear[k] || !mGrid.canCoarsen(parents[k])) continue;
        mGrid.coarsen(parents[k]);
        coarsened++;
    }
    phase.reset();
    if (refined + coarsened > 0) mGrid.synchronize();
    if (refined > 0 && mSplitOnRefine) splitOversizedSpringls();
}

void AdaptiveSpringLevelSet3D::adaptToParticles()
{
    const float margin = bandMargin();
    Points3 particles;
    particles.reserve(mManifold.springlParticles.size());
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (levelSetIncludes(n)) particles.push_back(mManifold.springlParticles[n]);
    }
    std::optional<ProfileScope> phase(std::in_place, "adaptToParticles: point grid");
    PointGrid3D grid;
    grid.build(particles, std::max(2.0f * margin, 1.0f));
    phase.reset();
    adaptBand([&](const Block &block) {
        const Vec3 lo = block.origin - Vec3::Constant(margin);
        return grid.anyInBox(lo, lo + Vec3::Constant(block.size() + 2.0f * margin));
    });
}

void AdaptiveSpringLevelSet3D::extractGridSurface()
{
    ProfileScope profileScope("extractGridSurface");
    mGrid.extractSurface(elementType(), mManifold.isoSurface, mContourScales);
    orientVertexNormals(mManifold.isoSurface);
}

void AdaptiveSpringLevelSet3D::evolveGrid()
{
    mGrid.updateActive();
    const Manifold3D &m = mManifold;
    TriangleTree3D springlTree;
    springlTree.reserve(m.springlVertexes.size());
    for (size_t n = 0; n < m.springlParticles.size(); n++) {
        if (!levelSetIncludes(n)) continue;
        const Vec3 *v = m.springl(n);
        const int K = m.springlSize(n);
        for (int k = 0; k < K; k++) springlTree.append(m.springlParticles[n], v[k], v[(k + 1) % K], static_cast<int>(n));
    }
    springlTree.finish();
    const float maxDistance = tuning.unsignedMaxDistance;
    // Pull toward the nearest springl: -min(U, 1.5 cells) * grad(U), in cell units.
    const auto pull = [&](const Vec3 &q, float h) -> Vec3 {
        const auto hit = springlTree.nearest(q, maxDistance);
        if (!hit || hit->distanceSqr <= 1E-18f) return Vec3::Zero();
        const float U = std::sqrt(hit->distanceSqr);
        return -std::min(U / h, kMaxPullCells) * (q - hit->point) / U;
    };
    mGrid.evolve(tuning.signedEvolveIterations, mCurvatureWeight, pull);
    if (adaptiveTuning.springlCorrection) {
        // Within a cell of a springl and over its element rather than past its outer edges (where
        // the side is ambiguous): signed by the springl's outward normal.
        mGrid.correctSigns([&](const Vec3 &q, float h, float &d) {
            const auto hit = springlTree.nearest(q, h);
            if (!hit) return false;
            const size_t s = static_cast<size_t>(hit->id);
            constexpr float kEdgeTolerance = 1E-10f;
            const Vec3 *v = m.springl(s);
            const int K = m.springlSize(s);
            for (int k = 0; k < K; k++) {
                if (SegmentDistanceSqr(v[k], v[(k + 1) % K], hit->point) < kEdgeTolerance) return false;
            }
            const float side = m.springlParticleNormals[s].dot(q - hit->point);
            if (side == 0.0f) return false;
            d = (side < 0.0f) ? -std::sqrt(hit->distanceSqr) : std::sqrt(hit->distanceSqr);
            return true;
        });
    }
    extractGridSurface();
}

void AdaptiveSpringLevelSet3D::evolveLevelSetAndContour()
{
    ProfileScope profileScope("level set: combined (+ adapt, refine)");
    // The band moves with the springls before the level set follows them.
    adaptToParticles();
    evolveGrid();
    refineContour(mManifold.isoSurface, tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
    mManifold.updateNormals();
}

void AdaptiveSpringLevelSet3D::splitSpringl(size_t index)
{
    Manifold3D &m = mManifold;
    const int K = m.springlSize(index);
    const Vec3 p = m.springlParticles[index];
    std::array<Vec3, kMaxSpringlSize> v;
    std::copy(m.springl(index), m.springl(index) + K, v.begin());
    const auto mid = [&](int a, int b) -> Vec3 { return 0.5f * (v[a % K] + v[b % K]); };
    // Children at half size, wound like the parent: a triangle splits into its four midpoint
    // triangles; any other polygon into one quad per corner (corner, edge midpoints, particle).
    std::vector<std::array<Vec3, 4>> children;
    int childSize = 4;
    if (K == 3) {
        childSize = 3;
        children = {{v[0], mid(0, 1), mid(2, 0), Vec3::Zero()},
                    {mid(0, 1), v[1], mid(1, 2), Vec3::Zero()},
                    {mid(2, 0), mid(1, 2), v[2], Vec3::Zero()},
                    {mid(0, 1), mid(1, 2), mid(2, 0), Vec3::Zero()}};
    } else {
        for (int c = 0; c < K; c++) children.push_back({v[c], mid(c, c + 1), p, mid(c + K - 1, c)});
    }
    const float half = 0.5f * springlScale(index);
    for (const auto &child : children) {
        const size_t other = SpringLevelSet3D::appendSpringl(Centroid(child.data(), childSize), std::span<const Vec3>(child.data(), childSize), half);
        m.velocities[other] = m.velocities[index];
        if (mTrackingEnabled) {
            m.correspondence[other] = m.correspondence[index];
            m.springlParticleTracking[other] = m.springlParticleTracking[index];
        }
    }
}

void AdaptiveSpringLevelSet3D::splitOversizedSpringls()
{
    ProfileScope profileScope("splitOversizedSpringls");
    // Children are appended (and visited too, so a springl keeps splitting until it fits its
    // leaf); split parents are removed at the end.
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    bool split = false;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (!(springlScale(n) > kSplitRatio * mGrid.cellSize(mManifold.springlParticles[n]))) continue;
        splitSpringl(n);
        keep.resize(mManifold.springlParticles.size(), 1);
        keep[n] = 0;
        split = true;
    }
    if (split) compactSpringls(keep);
    mManifold.updateNormals();
    mNearestNeighbors.clear();  // indexes changed; neighbors are found again before relaxing
}

// ---------------------------------------------------------------------------------------------
// Adaptive doubleweave
// ---------------------------------------------------------------------------------------------

AdaptiveDoubleweaveLevelSet3D::AdaptiveDoubleweaveLevelSet3D(ElementType elementType) : AdaptiveSpringLevelSet3D(elementType)
{
    mName = "Adaptive Doubleweave 3D";
}

size_t AdaptiveDoubleweaveLevelSet3D::countSpringls(Constellation constellation) const
{
    return static_cast<size_t>(std::count(mManifold.springlConstellations.begin(), mManifold.springlConstellations.end(), static_cast<uint8_t>(constellation)));
}

size_t AdaptiveDoubleweaveLevelSet3D::appendSpringl(const Vec3 &particle, std::span<const Vec3> vertexes, float scale, Constellation c)
{
    const size_t index = SpringLevelSet3D::appendSpringl(particle, vertexes, scale);
    mManifold.springlConstellations.push_back(static_cast<uint8_t>(c));
    return index;
}

void AdaptiveDoubleweaveLevelSet3D::seedSpringls()
{
    const SurfaceMesh3D &iso = mManifold.isoSurface;
    const int K = elementSize();
    std::array<Vec3, kMaxElementSize> verts;
    for (size_t f = 0; f < iso.faceCount(); f++) {
        float scale = std::numeric_limits<float>::max();
        for (int k = 0; k < K; k++) {
            verts[k] = iso.vertexes[iso.face(f)[k]];
            scale = std::min(scale, contourScale(iso.face(f)[k]));
        }
        appendSpringl(Centroid(verts.data(), K), std::span<const Vec3>(verts.data(), K), scale, Constellation::Primal);
    }
    ForEachDualSpringl(
        iso, [&](const Vec3 &particle, std::span<const Vec3> v, uint32_t vertex) { appendSpringl(particle, v, contourScale(vertex), Constellation::Dual); });
}

void AdaptiveDoubleweaveLevelSet3D::clearSpringls()
{
    AdaptiveSpringLevelSet3D::clearSpringls();
    mManifold.springlConstellations.clear();
}

void AdaptiveDoubleweaveLevelSet3D::compactSpringls(const std::vector<uint8_t> &keep)
{
    AdaptiveSpringLevelSet3D::compactSpringls(keep);
    size_t out = 0;
    for (size_t i = 0; i < keep.size(); i++) {
        if (keep[i]) mManifold.springlConstellations[out++] = mManifold.springlConstellations[i];
    }
    mManifold.springlConstellations.resize(out);
}

void AdaptiveDoubleweaveLevelSet3D::snapshotForTracking()
{
    AdaptiveSpringLevelSet3D::snapshotForTracking();
    mOldConstellations = mManifold.springlConstellations;
}

bool AdaptiveDoubleweaveLevelSet3D::canTrackFrom(size_t newIndex, size_t oldIndex) const
{
    return newIndex < mManifold.springlConstellations.size() && oldIndex < mOldConstellations.size() &&
           mManifold.springlConstellations[newIndex] == mOldConstellations[oldIndex];
}

bool AdaptiveDoubleweaveLevelSet3D::levelSetIncludes(size_t springl) const
{
    return mActiveConstellation < 0 ||
           (springl < mManifold.springlConstellations.size() && mManifold.springlConstellations[springl] == static_cast<uint8_t>(mActiveConstellation));
}

void AdaptiveDoubleweaveLevelSet3D::splitSpringl(size_t index)
{
    // Same split as the base class; the appended children keep the constellation.
    const uint8_t c = mManifold.springlConstellations[index];
    AdaptiveSpringLevelSet3D::splitSpringl(index);
    mManifold.springlConstellations.resize(mManifold.springlParticles.size(), c);
}

void AdaptiveDoubleweaveLevelSet3D::swapState(size_t c)
{
    ConstellationState &state = mStates[c];
    std::swap(mGrid, state.grid);
    std::swap(mManifold.isoSurface, state.isoSurface);
    std::swap(mContourScales, state.contourScales);
}

bool AdaptiveDoubleweaveLevelSet3D::init()
{
    mActiveConstellation = -1;
    if (!AdaptiveSpringLevelSet3D::init()) return false;
    // Each constellation starts from the combined level set, then follows its own springls.
    for (ConstellationState &state : mStates) {
        state.grid = mGrid;
        state.isoSurface = mManifold.isoSurface;
        state.contourScales = mContourScales;
    }
    evolveConstellations();
    return true;
}

void AdaptiveDoubleweaveLevelSet3D::evolveConstellations()
{
    ProfileScope profileScope("level set: constellations");
    for (size_t c = 0; c < kConstellationCount; c++) {
        swapState(c);
        mActiveConstellation = static_cast<int>(c);
        mSplitOnRefine = false;  // springls follow the combined grid only
        adaptToParticles();
        evolveGrid();
        refineContour(mManifold.isoSurface, tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
        mSplitOnRefine = true;
        mActiveConstellation = -1;
        swapState(c);
    }
}

void AdaptiveDoubleweaveLevelSet3D::evolveLevelSetAndContour()
{
    AdaptiveSpringLevelSet3D::evolveLevelSetAndContour();  // combined grid: output and springl splitting
    evolveConstellations();
}

int AdaptiveDoubleweaveLevelSet3D::contract()
{
    ProfileScope profileScope("contract");
    // Each constellation against its own iso-surface.
    std::vector<uint8_t> keep(mManifold.springlParticles.size(), 1);
    int contracted = 0;
    for (size_t c = 0; c < kConstellationCount; c++) {
        swapState(c);
        contracted += markContractions(keep, [&](size_t n) { return mManifold.springlConstellations[n] == c; });
        swapState(c);
    }
    if (contracted > 0) compactSpringls(keep);
    return contracted;
}

int AdaptiveDoubleweaveLevelSet3D::fill()
{
    ProfileScope profileScope("fill");
    // Gap tests per constellation against the springls that exist before this pass; each
    // constellation fills from its own iso-surface.
    std::optional<ProfileScope> phase(std::in_place, "fill: refresh springl grid");
    refreshSpringlGrid();
    phase.reset();
    const int K = elementSize();
    using Candidate = FillCandidate;
    std::array<std::vector<Candidate>, kConstellationCount> gaps;
    // The candidates come from the iso-surfaces, which only change when the level sets do (once per
    // sub-step), so the fill passes of one sub-step share them.
    const bool fresh = mFillCandidatesStamp != mAdvectionSubstepCount;
    mFillCandidatesStamp = mAdvectionSubstepCount;
    for (size_t w = 0; w < kConstellationCount; w++) {
        std::vector<Candidate> &candidates = mFillCandidates[w];
        if (fresh) {
            swapState(w);
            phase.emplace("fill: candidates");
            const SurfaceMesh3D &iso = mManifold.isoSurface;
            candidates.clear();
            if (w == static_cast<size_t>(Constellation::Primal)) {
                candidates.resize(iso.faceCount());
                for (size_t f = 0; f < iso.faceCount(); f++) {
                    Candidate &c = candidates[f];
                    c.scale = std::numeric_limits<float>::max();
                    for (int k = 0; k < K; k++) {
                        c.vertexes[k] = iso.vertexes[iso.face(f)[k]];
                        c.scale = std::min(c.scale, contourScale(iso.face(f)[k]));
                    }
                    c.size = K;
                    c.particle = Centroid(c.vertexes.data(), K);
                }
            } else {
                ForEachDualSpringl(iso, [&](const Vec3 &particle, std::span<const Vec3> v, uint32_t vertex) {
                    Candidate c;
                    c.particle = particle;
                    std::copy(v.begin(), v.end(), c.vertexes.begin());
                    c.size = static_cast<int>(v.size());
                    c.scale = contourScale(vertex);
                    candidates.push_back(c);
                });
            }
            swapState(w);
        }
        phase.emplace("fill: gap tests");
        const int C = static_cast<int>(candidates.size());
        const uint8_t tag = static_cast<uint8_t>(w);
        const std::vector<uint8_t> &constellations = mManifold.springlConstellations;
        std::vector<uint8_t> isGap(C, 0);
        SPRINGLS_OMP(omp parallel for if (C >= kParallelMinHeavyItems))
        for (int c = 0; c < C; c++) {
            isGap[c] = !mSpringlGrid.anyElementWithin(mManifold.springlParticles, mManifold.springlVertexes, candidates[c].particle,
                                                      tuning.fillDistance * candidates[c].scale, [&](size_t n) { return constellations[n] == tag; });
        }
        for (int c = 0; c < C; c++) {
            if (isGap[c]) gaps[w].push_back(candidates[c]);
        }
    }
    phase.emplace("fill: append");
    int filled = 0;
    for (size_t w = 0; w < kConstellationCount; w++) {
        for (const Candidate &c : gaps[w]) {
            appendSpringl(c.particle, std::span<const Vec3>(c.vertexes.data(), c.size), c.scale, static_cast<Constellation>(w));
            filled++;
        }
    }
    return filled;
}
}  // namespace springls
