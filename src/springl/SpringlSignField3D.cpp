#include <core/Common.hpp>
#include <core/Profile.hpp>
#include <springl/SpringlSignField3D.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace springls
{
namespace
{
const std::array<Vec3, 6> kAxes = {Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1)};
}  // namespace

int SpringlSignField3D::latticeAt(const Vec3 &p) const
{
    const int64_t i = std::llround(p.x() / mH) - mLatticeLo[0], j = std::llround(p.y() / mH) - mLatticeLo[1], k = std::llround(p.z() / mH) - mLatticeLo[2];
    if (i < 0 || j < 0 || k < 0 || i >= mLatticeDims.x() || j >= mLatticeDims.y() || k >= mLatticeDims.z()) return -1;
    return mLatticeIndex[(k * mLatticeDims.y() + j) * mLatticeDims.x() + i];
}

void SpringlSignField3D::build(const SignFieldInput3D &input, const SignFieldParams3D &params)
{
    ProfileScope profileScope("sign field: build");
    mInput = input;
    mParams = params;
    const AdaptiveGrid3D &grid = *mInput.grid;
    const auto &blocks = grid.blocks();
    const Points3 &particles = *mInput.particles;
    const Points3 &vertexes = *mInput.vertexes;
    const std::vector<uint32_t> &start = *mInput.start;

    std::optional<ProfileScope> phase(std::in_place, "sign field: tree + lattice");
    mTree.reset();
    for (uint32_t n : mInput.springls) {
        const uint32_t begin = start[n];
        const int K = static_cast<int>(start[n + 1] - begin);
        for (int k = 0; k < K; k++) {
            mTree.append(particles[n], vertexes[begin + k], vertexes[begin + (k + 1) % K], static_cast<int>(n));
        }
    }
    mTree.finish();

    // Finest cell size, and the lattice of the finest leaves' nodes.
    mH = std::numeric_limits<float>::infinity();
    for (int leaf : grid.leaves()) mH = std::min(mH, blocks[leaf].h);
    if (!std::isfinite(mH)) mH = 1.0f;
    mBandRadius = std::max({mParams.capsuleCells * mH, kMinBandRadius * mH, mParams.minBandVoxels});
    const float finestTolerance = 1.001f * mH;
    mPositions.clear();
    mCoarseNode.assign(blocks.size(), -1);
    std::vector<int> coarseLeaves, finestLeaves;
    int64_t lo[3] = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()};
    int64_t hi[3] = {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min()};
    const auto latticeOrigin = [&](const AdaptiveGrid3D::Block &block, int axis) { return static_cast<int64_t>(std::llround(block.origin[axis] / mH)); };
    for (int leaf : grid.leaves()) {
        const AdaptiveGrid3D::Block &block = blocks[leaf];
        if (block.h > finestTolerance) {
            coarseLeaves.push_back(leaf);
            continue;
        }
        finestLeaves.push_back(leaf);
        for (int axis = 0; axis < 3; axis++) {
            lo[axis] = std::min(lo[axis], latticeOrigin(block, axis));
            hi[axis] = std::max(hi[axis], latticeOrigin(block, axis) + AdaptiveGrid3D::kBlockCells);
        }
    }
    // Dense index over the finest leaves' bounding box: lattice lookups are one array read.
    if (finestLeaves.empty()) {
        mLatticeDims = Vec3i::Zero();
        mLatticeIndex.clear();
    } else {
        for (int axis = 0; axis < 3; axis++) mLatticeLo[axis] = lo[axis];
        mLatticeDims = Vec3i(static_cast<int>(hi[0] - lo[0] + 1), static_cast<int>(hi[1] - lo[1] + 1), static_cast<int>(hi[2] - lo[2] + 1));
        mLatticeIndex.assign(static_cast<size_t>(mLatticeDims.x()) * mLatticeDims.y() * mLatticeDims.z(), -1);
    }
    // Mark the finest leaves' nodes (in parallel; every writer stores the same value), then number
    // them slice by slice: counts per z slice, a prefix sum, and the slices numbered in parallel.
    {
        const int F = static_cast<int>(finestLeaves.size());
        SPRINGLS_OMP(omp parallel for if (F >= kParallelMinBlocks))
        for (int f = 0; f < F; f++) {
            const AdaptiveGrid3D::Block &block = blocks[finestLeaves[f]];
            const int64_t i0 = latticeOrigin(block, 0) - mLatticeLo[0], j0 = latticeOrigin(block, 1) - mLatticeLo[1],
                          k0 = latticeOrigin(block, 2) - mLatticeLo[2];
            for (int c = 0; c < AdaptiveGrid3D::kBlockNodes; c++) {
                for (int b = 0; b < AdaptiveGrid3D::kBlockNodes; b++) {
                    for (int a = 0; a < AdaptiveGrid3D::kBlockNodes; a++) {
                        std::atomic_ref<int>(mLatticeIndex[((k0 + c) * mLatticeDims.y() + (j0 + b)) * mLatticeDims.x() + (i0 + a)])
                            .store(0, std::memory_order_relaxed);
                    }
                }
            }
        }
        const int slices = mLatticeDims.z();
        const size_t sliceSize = static_cast<size_t>(mLatticeDims.x()) * mLatticeDims.y();
        std::vector<int> sliceStart(static_cast<size_t>(slices) + 1, 0);
        SPRINGLS_OMP(omp parallel for if (slices >= 8))
        for (int k = 0; k < slices; k++) {
            int count = 0;
            for (size_t i = 0; i < sliceSize; i++) count += mLatticeIndex[k * sliceSize + i] >= 0;
            sliceStart[k + 1] = count;
        }
        for (int k = 0; k < slices; k++) sliceStart[k + 1] += sliceStart[k];
        mPositions.resize(static_cast<size_t>(sliceStart[slices]));
        const int W = mLatticeDims.x(), Hy = mLatticeDims.y();
        SPRINGLS_OMP(omp parallel for if (slices >= 8))
        for (int k = 0; k < slices; k++) {
            int id = sliceStart[k];
            for (int j = 0; j < Hy; j++) {
                for (int i = 0; i < W; i++) {
                    int &slot = mLatticeIndex[k * sliceSize + static_cast<size_t>(j) * W + i];
                    if (slot < 0) continue;
                    slot = id;
                    mPositions[id++] =
                        mH * Vec3(static_cast<float>(mLatticeLo[0] + i), static_cast<float>(mLatticeLo[1] + j), static_cast<float>(mLatticeLo[2] + k));
                }
            }
        }
    }
    const int L = static_cast<int>(mPositions.size());
    const int C = static_cast<int>(coarseLeaves.size());
    for (int n = 0; n < C; n++) mCoarseNode[coarseLeaves[n]] = L + n;
    mCoarseSign.assign(blocks.size(), 1);
    mWall.assign(L, 0);
    mSign.assign(L, 1);
    mDistance.assign(L, mParams.distanceClamp);
    mWallCount = 0;
    phase.reset();
    if (mTree.empty()) {
        // No springls: no surface.
        return;  // outside everywhere (mSign and mCoarseSign start at +1)
    }

    const float R = mBandRadius;
    const float clamp = mParams.distanceClamp;

    // 1. Unsigned distance U: the min over the springls (distance to the union of their elements).
    // Walls: the springls' capsules, the nodes with U <= R.
    phase.emplace("sign field: unsigned distance");
    std::vector<float> U(L, std::numeric_limits<float>::infinity());
    const float reach = std::max(R, clamp);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 1024) if (L >= kParallelMinHeavyItems))
    for (int n = 0; n < L; n++) {
        const auto hit = mTree.nearest(mPositions[n], std::nextafter(reach, std::numeric_limits<float>::infinity()));
        if (!hit) continue;
        U[n] = std::sqrt(hit->distanceSqr);
        if (U[n] <= R) mWall[n] = 1;
    }
    for (int n = 0; n < L; n++) mWallCount += mWall[n];

    // 2. The exterior floods in from the domain border around the walls; non-wall nodes it does not
    // reach are inside.
    phase.reset();
    std::vector<int> neighbors;
    std::vector<uint8_t> exterior;
    floodExterior(coarseLeaves, neighbors, exterior);
    for (int n = 0; n < L; n++) {
        if (!mWall[n]) mSign[n] = exterior[n] ? 1 : -1;
    }
    for (int c = 0; c < C; c++) mCoarseSign[coarseLeaves[c]] = exterior[L + c] ? 1 : -1;

    // 3. Upwind sign into the walls (see the class comment).
    phase.emplace("sign field: upwind into walls");
    // Wall nodes in order of decreasing U (ties by index), sorted in parallel: bucketed by U, each
    // bucket sorted on its own, concatenated (the same order as one sort).
    std::vector<int> wall;
    for (int n = 0; n < L; n++) {
        if (mWall[n]) wall.push_back(n);
    }
    {
        constexpr int kBuckets = 256;
        const auto bucketOf = [&](int n) { return std::clamp(static_cast<int>((1.0f - U[n] / R) * kBuckets), 0, kBuckets - 1); };
        std::vector<std::vector<int>> buckets(kBuckets);
        for (int n : wall) buckets[bucketOf(n)].push_back(n);
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 1) if (wall.size() >= static_cast<size_t>(kParallelMinHeavyItems)))
        for (int k = 0; k < kBuckets; k++) {
            std::sort(buckets[k].begin(), buckets[k].end(), [&](int x, int y) { return U[x] != U[y] ? U[x] > U[y] : x < y; });
        }
        wall.clear();
        for (const auto &bucket : buckets) wall.insert(wall.end(), bucket.begin(), bucket.end());
    }
    // Each wall node takes the sign of its neighbors with larger U, weighted by how much larger, so
    // the outside and inside fronts meet at U = 0. A node with no such neighbor (the middle of a
    // sheet thinner than two capsules) is inside.
    std::vector<uint8_t> assigned(L, 0);
    for (int n = 0; n < L; n++) assigned[n] = !mWall[n];
    for (int n : wall) {
        float vote = 0.0f;
        bool any = false;
        for (int a = 0; a < 6; a++) {
            const int m = neighbors[6 * static_cast<size_t>(n) + a];
            if (m < 0) continue;
            if (m >= L) {
                // A coarse leaf beside a wall node lies beyond the band: upwind by construction.
                vote += static_cast<float>(mCoarseSign[coarseLeaves[m - L]]) * mH;
                any = true;
                continue;
            }
            if (!assigned[m] || U[m] <= U[n]) continue;
            const float u = std::isfinite(U[m]) ? U[m] : reach;
            vote += static_cast<float>(mSign[m]) * (u - U[n]);
            any = true;
        }
        if (any && vote != 0.0f) {
            mSign[n] = vote < 0.0f ? -1 : 1;
        } else {
            mSign[n] = -1;  // the middle of a feature thinner than two capsules: inside
        }
        assigned[n] = 1;
    }

    // 4. |phi| = min(U, clamp).
    for (int n = 0; n < L; n++) mDistance[n] = std::min(U[n], clamp);
}

void SpringlSignField3D::floodExterior(const std::vector<int> &coarseLeaves, std::vector<int> &neighbors, std::vector<uint8_t> &exterior) const
{
    ProfileScope profileScope("sign field: flood exterior (+graph)");
    const AdaptiveGrid3D &grid = *mInput.grid;
    const auto &blocks = grid.blocks();
    const int L = static_cast<int>(mPositions.size());
    const int C = static_cast<int>(coarseLeaves.size());
    const Vec3 domainMax = grid.domainMax();
    // Region graph: lattice nodes (6-neighbors) and coarse leaves (joined across faces).
    const auto neighborNode = [&](int n, int a) -> int {
        const Vec3 x = mPositions[n] + mH * kAxes[a];
        const int m = latticeAt(x);
        if (m >= 0) return m;
        if ((x.array() < 0.0f).any() || (x.array() > domainMax.array()).any()) return -1;
        const int leaf = grid.findLeaf(x);
        return isCoarseLeaf(leaf) ? mCoarseNode[leaf] : -1;
    };
    neighbors.assign(static_cast<size_t>(6) * L, -1);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4096) if (L >= kParallelMinHeavyItems))
    for (int n = 0; n < L; n++) {
        for (int a = 0; a < 6; a++) neighbors[6 * static_cast<size_t>(n) + a] = neighborNode(n, a);
    }
    // Coarse leaves: neighbors across faces (probes just past each face center, both directions),
    // and lattice nodes that step into them.
    std::vector<std::vector<int>> coarseAdjacent(C);
    for (int c = 0; c < C; c++) {
        const AdaptiveGrid3D::Block &block = blocks[coarseLeaves[c]];
        const Vec3 center = block.origin + Vec3::Constant(0.5f * block.size());
        for (int a = 0; a < 6; a++) {
            const Vec3 x = center + (0.5f * block.size() + 0.25f * mH) * kAxes[a];
            if ((x.array() < 0.0f).any() || (x.array() > domainMax.array()).any()) continue;
            const int leaf = grid.findLeaf(x);
            if (!isCoarseLeaf(leaf)) continue;
            const int other = mCoarseNode[leaf] - L;
            coarseAdjacent[c].push_back(L + other);
            coarseAdjacent[other].push_back(L + c);
        }
    }
    for (int n = 0; n < L; n++) {
        for (int a = 0; a < 6; a++) {
            const int m = neighbors[6 * static_cast<size_t>(n) + a];
            if (m >= L) coarseAdjacent[m - L].push_back(n);
        }
    }

    const float edge = 0.5f * mH;
    // Breadth-first, frontier by frontier in parallel: a node joins the exterior when an atomic
    // exchange on its flag succeeds, so the reached set is the same whatever the thread order.
    exterior.assign(L + C, 0);
    const auto claim = [&](int n) { return (n >= L || !mWall[n]) && std::atomic_ref<uint8_t>(exterior[n]).exchange(1, std::memory_order_relaxed) == 0; };
    std::vector<int> frontier;
    for (int n = 0; n < L; n++) {
        const Vec3 &q = mPositions[n];
        if (((q.array() <= edge).any() || (q.array() >= domainMax.array() - edge).any()) && claim(n)) frontier.push_back(n);
    }
    for (int c = 0; c < C; c++) {
        const AdaptiveGrid3D::Block &block = blocks[coarseLeaves[c]];
        const Vec3 hi = block.origin + Vec3::Constant(block.size());
        if (((block.origin.array() <= edge).any() || (hi.array() >= domainMax.array() - edge).any()) && claim(L + c)) frontier.push_back(L + c);
    }
    std::vector<int> next;
    while (!frontier.empty()) {
        next.clear();
        const int F = static_cast<int>(frontier.size());
        SPRINGLS_OMP(omp parallel if (F >= kParallelMinHeavyItems))
        {
            std::vector<int> local;
            SPRINGLS_OMP(omp for schedule(dynamic, 1024) nowait)
            for (int k = 0; k < F; k++) {
                const int n = frontier[k];
                if (n < L) {
                    for (int a = 0; a < 6; a++) {
                        const int m = neighbors[6 * static_cast<size_t>(n) + a];
                        if (m >= 0 && claim(m)) local.push_back(m);
                    }
                } else {
                    for (int m : coarseAdjacent[n - L]) {
                        if (claim(m)) local.push_back(m);
                    }
                }
            }
            SPRINGLS_OMP(omp critical)
            next.insert(next.end(), local.begin(), local.end());
        }
        frontier.swap(next);
    }
}

float SpringlSignField3D::distance(const Vec3 &p, float maxDistance) const
{
    if (mTree.empty()) return maxDistance;
    const int n = latticeAt(p);
    if (n >= 0 && maxDistance <= mParams.distanceClamp && (mPositions[n] - p).squaredNorm() < 1E-4f * mH * mH) return std::min(mDistance[n], maxDistance);
    return std::min(mTree.distance(p, maxDistance), maxDistance);
}

float SpringlSignField3D::sign(const Vec3 &p) const
{
    const int n = latticeAt(p);
    if (n >= 0) return mSign[n];
    const int leaf = mInput.grid->findLeaf(p);
    if (isCoarseLeaf(leaf)) return mCoarseSign[leaf];
    return 1.0f;
}

void FillGridFromSignField(AdaptiveGrid3D &grid, const SpringlSignField3D &field, float band, float clamp)
{
    using Block = AdaptiveGrid3D::Block;
    ProfileScope profileScope("rebuild: phi at nodes");

    // phi = sign * min(U, clamp) on the finest leaves, where the surface lies: the distance is only
    // needed a few cells out (contour interpolation and gradients near the surface), and the clamp
    // keeps each closest-triangle query local. Coarser leaves never hold the surface; they are
    // tiles of +-band with their region's sign, as in the CFL-limited grid.
    const float finest = field.latticeSpacing();
    const std::vector<int> &leaves = grid.leaves();
    const int L = static_cast<int>(leaves.size());
    std::vector<std::vector<float>> values(leaves.size());
    std::vector<float> tileValue(leaves.size(), 0.0f);
    {
        const auto &blocks = grid.blocks();
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (L >= kParallelMinBlocks))
        for (int n = 0; n < L; n++) {
            const Block &block = blocks[leaves[n]];
            const Vec3 center = block.origin + Vec3::Constant(0.5f * block.size());
            const float halfDiagonal = 0.5f * std::sqrt(3.0f) * block.size();
            if (block.h > 1.001f * finest || !field.anyWithin(center, halfDiagonal + clamp)) {
                tileValue[n] = field.sign(center) * band;
                continue;
            }
            std::vector<float> &v = values[n];
            v.resize(AdaptiveGrid3D::kNodeCount);
            for (int c = 0; c < AdaptiveGrid3D::kBlockNodes; c++) {
                for (int b = 0; b < AdaptiveGrid3D::kBlockNodes; b++) {
                    for (int a = 0; a < AdaptiveGrid3D::kBlockNodes; a++) v[Block::index(a, b, c)] = field.phi(block.nodePosition(a, b, c), clamp);
                }
            }
        }
    }
    auto &blocks = grid.blocks();
    for (int n = 0; n < L; n++) {
        Block &block = blocks[leaves[n]];
        const std::vector<float> &v = values[n];
        if (v.empty()) {
            block.makeTile(tileValue[n]);
            continue;
        }
        const float first = v[0];
        if (std::abs(first) == clamp && std::all_of(v.begin(), v.end(), [first](float x) { return x == first; })) {
            block.makeTile(first > 0.0f ? band : -band);
        } else {
            block.phi = v;
        }
    }
}
}  // namespace springls
