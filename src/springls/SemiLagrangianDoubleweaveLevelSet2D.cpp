#include <springls/SemiLagrangianDoubleweaveLevelSet2D.hpp>

#include <springls/SpringlSignField2D.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace springls
{
namespace
{
// Without a schedule (or when nothing else bounds the step) a sub-step is at most this long.
constexpr float kFallbackMaxTimeStep = 1.0f;
}  // namespace

SemiLagrangianDoubleweaveLevelSet2D::SemiLagrangianDoubleweaveLevelSet2D() { mName = "Semi-Lagrangian Doubleweave"; }

bool SemiLagrangianDoubleweaveLevelSet2D::step()
{
    // The strain estimate reads attraction neighbors; merging springls after a step clears them.
    if (mNearestNeighbor.size() != mManifold.springlVertexes.size()) updateNearestNeighbors();
    return AdaptiveDoubleweaveLevelSet2D::step();
}

void SemiLagrangianDoubleweaveLevelSet2D::evolveGrid()
{
    mManifold.updateNormals();
    refreshSpringlGrid();
    // The field keeps pointers to the springl arrays and reads mGrid through the callbacks, from
    // OpenMP threads: neither is modified until every node has been sampled.
    SignFieldInput input;
    const Vec2 domain = mGrid.domainMax();
    input.width = static_cast<int>(std::ceil(domain.x())) + 1;
    input.height = static_cast<int>(std::ceil(domain.y())) + 1;
    input.vertexes = &mManifold.springlVertexes;
    for (size_t n = 0; n < mManifold.springlParticles.size(); n++) {
        if (levelSetIncludes(n)) input.springls.push_back(static_cast<uint32_t>(n));
    }
    // Raster at the finest cell size in the grid, so sub-pixel features of refined leaves survive.
    float finest = std::numeric_limits<float>::infinity();
    for (int leaf : mGrid.leaves()) finest = std::min(finest, mGrid.blocks()[leaf].h);
    input.rasterSpacing = std::isfinite(finest) ? finest : 1.0f;
    const AdaptiveGrid2D &grid = mGrid;
    input.cellSize = [&grid](const Vec2 &p) { return grid.cellSize(p); };
    SignFieldParams params;
    params.capsuleCells = slTuning.capsuleCells;
    params.minBandPixels = slTuning.minBandPixels;
    SpringlSignField2D field;
    field.build(input, params);

    const float band = adaptiveTuning.bandWidth;
    const std::vector<int> &leaves = mGrid.leaves();
    const int L = static_cast<int>(leaves.size());
    std::vector<std::array<float, AdaptiveGrid2D::kNodeCount>> values(leaves.size());
    {
        const auto &blocks = mGrid.blocks();
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (L >= kParallelMinBlocks))
        for (int n = 0; n < L; n++) {
            const AdaptiveGrid2D::Block &block = blocks[leaves[n]];
            for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
                for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) {
                    values[n][AdaptiveGrid2D::Block::index(a, b)] = field.phi(block.nodePosition(a, b), band);
                }
            }
        }
    }
    auto &blocks = mGrid.blocks();
    for (int n = 0; n < L; n++) {
        AdaptiveGrid2D::Block &block = blocks[leaves[n]];
        const auto &v = values[n];
        const float first = v[0];
        const bool uniform = std::abs(first) == band && std::all_of(v.begin(), v.end(), [first](float x) { return x == first; });
        if (uniform) {
            block.makeTile(first);
        } else {
            block.materialize();
            std::copy(v.begin(), v.end(), block.phi.begin());
        }
    }
    mGrid.synchronize();
    mGrid.updateActive();
    extractGridContour();
    if (eraseTinyLoops() > 0) {
        mGrid.synchronize();
        mGrid.updateActive();
        extractGridContour();
    }
}

int SemiLagrangianDoubleweaveLevelSet2D::eraseTinyLoops()
{
    const Points2 &vertexes = mManifold.vertexLocations;
    auto &blocks = mGrid.blocks();
    int erased = 0;
    for (const Curve &curve : mManifold.vertexIndexes) {
        if (curve.size() < 4 || curve.front() != curve.back()) continue;  // open, or degenerate
        double area = 0.0;
        float h = std::numeric_limits<float>::infinity();
        Vec2 lo = Vec2::Constant(std::numeric_limits<float>::infinity()), hi = Vec2::Constant(-std::numeric_limits<float>::infinity());
        for (size_t k = 0; k + 1 < curve.size(); k++) {
            const Vec2 a = vertexes[curve[k]], b = vertexes[curve[k + 1]];
            area += static_cast<double>(a.x()) * b.y() - static_cast<double>(b.x()) * a.y();
            h = std::min(h, contourScale(curve[k]));
            lo = lo.cwiseMin(a);
            hi = hi.cwiseMax(a);
        }
        if (std::abs(0.5 * area) >= static_cast<double>(slTuning.minLoopCells) * h * h) continue;
        // Nodes enclosed by the loop (even-odd rule).
        const auto enclosed = [&](const Vec2 &p) {
            bool inside = false;
            for (size_t k = 0; k + 1 < curve.size(); k++) {
                const Vec2 a = vertexes[curve[k]], b = vertexes[curve[k + 1]];
                if (((a.y() > p.y()) != (b.y() > p.y())) && (p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x())) inside = !inside;
            }
            return inside;
        };
        const auto forEnclosedNode = [&](auto &&visit) {
            for (int leaf : mGrid.leaves()) {
                AdaptiveGrid2D::Block &block = blocks[leaf];
                if (block.origin.x() > hi.x() || block.origin.y() > hi.y() || block.origin.x() + block.size() < lo.x() ||
                    block.origin.y() + block.size() < lo.y()) {
                    continue;
                }
                for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
                    for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) {
                        const Vec2 q = block.nodePosition(a, b);
                        if (q.x() < lo.x() || q.y() < lo.y() || q.x() > hi.x() || q.y() > hi.y() || !enclosed(q)) continue;
                        visit(block, a, b);
                    }
                }
            }
        };
        // The loop's interior sign is the majority sign of the nodes it encloses.
        int vote = 0;
        forEnclosedNode([&](const AdaptiveGrid2D::Block &block, int a, int b) { vote += block.at(a, b) < 0.0f ? -1 : 1; });
        if (vote == 0) continue;
        const bool interiorInside = vote < 0;
        // Flip the interior-side nodes to the outside sign, keeping |phi| = min(U, bandWidth).
        forEnclosedNode([&](AdaptiveGrid2D::Block &block, int a, int b) {
            const float v = static_cast<const AdaptiveGrid2D::Block &>(block).at(a, b);
            if ((v < 0.0f) == interiorInside) block.at(a, b) = -v;
        });
        erased++;
    }
    return erased;
}

float SemiLagrangianDoubleweaveLevelSet2D::advectionTimeStep(const Points2 &k1, float) const
{
    const size_t N = mManifold.springlParticles.size();
    const Points2 &vertexes = mManifold.springlVertexes;
    const Points2 &particles = mManifold.springlParticles;
    const double t = mSimulationTime;
    const float maxStepCells = slTuning.maxStepCells;
    // Neighbor pairs need stage-1 velocities for every springl and current neighbor links.
    const bool pairs = k1.size() == N && mNearestNeighbor.size() == 2 * N;
    // Primal and dual constellations do not attract each other, so their divergence is measured
    // separately: each particle against the particle of the nearest other-constellation springl.
    const bool crossPairs = k1.size() == N && mManifold.springlConstellations.size() == N;
    SpringlHashGrid2D springlGrid;
    if (crossPairs) springlGrid.update(particles, vertexes);
    const float crossReach = 2.0f * tuning.maxArmLength;
    // Only same-facing pairs measure stretching: opposite-facing pairs are fronts colliding (or a
    // filament's two sides), whose relative speed says nothing about how far the springls spread.
    const auto sameFacing = [&](size_t a, size_t b) { return (vertexes[2 * a + 1] - vertexes[2 * a]).dot(vertexes[2 * b + 1] - vertexes[2 * b]) > 0.0f; };
    const auto pairStrain = [&](size_t i, size_t j) {
        // Floored at a quarter springl: neighbors (and primal / dual particles) may nearly coincide.
        const float gap = std::max(distance(particles[i], particles[j]), kMinPairGapScale * springlScale(i));
        return (k1[i] - k1[j]).norm() / gap;
    };
    float maxStrain = 0.0f;
    float timeStep = std::numeric_limits<float>::infinity();
    const int count = static_cast<int>(N);
    SPRINGLS_OMP(omp parallel for reduction(max : maxStrain) reduction(min : timeStep) if (count >= kParallelMinHeavyItems))
    for (int i = 0; i < count; i++) {
        const size_t s = static_cast<size_t>(i);
        // Stretch within the springl...
        const Vec2 v0 = vertexes[2 * s], v1 = vertexes[2 * s + 1];
        const float length = distance(v0, v1);
        if (length > 0.0f) maxStrain = std::max(maxStrain, (particleVelocity(v1, t) - particleVelocity(v0, t)).norm() / length);
        // ...between attraction neighbors...
        if (pairs) {
            for (int e = 0; e < 2; e++) {
                const int neighbor = mNearestNeighbor[2 * s + e];
                if (neighbor < 0) continue;
                const size_t j = static_cast<size_t>(neighbor) / 2;
                if (j != s && sameFacing(s, j)) maxStrain = std::max(maxStrain, pairStrain(s, j));
            }
        }
        // ...and against the nearest springl of the other constellation.
        if (crossPairs) {
            const uint8_t mine = mManifold.springlConstellations[s];
            const int v = springlGrid.nearestVertex(vertexes, particles[s], crossReach * springlScale(s), s,
                                                    [&](size_t m) { return mManifold.springlConstellations[m] != mine && sameFacing(s, m); });
            if (v >= 0) maxStrain = std::max(maxStrain, pairStrain(s, static_cast<size_t>(v) / 2));
        }
        const float speed = (s < k1.size()) ? k1[s].norm() : 0.0f;
        if (speed > 0.0f && std::isfinite(maxStepCells)) timeStep = std::min(timeStep, maxStepCells * springlScale(s) / speed);
    }
    if (maxStrain > 0.0f) timeStep = std::min(timeStep, slTuning.maxStrain / maxStrain);
    if (!mSchedule.weight) timeStep = std::min(timeStep, kFallbackMaxTimeStep);
    // Nothing bounded the step (no strain, no maxStepCells), e.g. a rigid translation: fall back to
    // the unscheduled cap, or advect would move by inf. advect() still applies the schedule's cap.
    if (!std::isfinite(timeStep)) timeStep = kFallbackMaxTimeStep;
    return timeStep;
}

size_t SemiLagrangianDoubleweaveLevelSet2D::dropDegenerateSpringls(size_t from)
{
    const size_t N = mManifold.springlParticles.size();
    std::vector<uint8_t> keep(N, 1);
    size_t dropped = 0;
    for (size_t n = from; n < N; n++) {
        if (distance(mManifold.springlVertexes[2 * n], mManifold.springlVertexes[2 * n + 1]) < kMinSeedLengthCells * springlScale(n)) {
            keep[n] = 0;
            dropped++;
        }
    }
    if (dropped > 0) {
        compactSpringls(keep);
        mManifold.updateNormals();
    }
    return dropped;
}

void SemiLagrangianDoubleweaveLevelSet2D::seedSpringls()
{
    AdaptiveDoubleweaveLevelSet2D::seedSpringls();
    dropDegenerateSpringls(0);
}

int SemiLagrangianDoubleweaveLevelSet2D::fill()
{
    const size_t before = mManifold.springlParticles.size();
    const int added = AdaptiveDoubleweaveLevelSet2D::fill();
    return added - static_cast<int>(dropDegenerateSpringls(before));
}
}  // namespace springls
