#include <core/Profile.hpp>
#include <mesh/MeshSplat3D.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace springls
{
namespace
{
using Block = AdaptiveGrid3D::Block;
constexpr int N = AdaptiveGrid3D::kBlockCells;

// Lattice coordinates (cells of `size`, count[a] per axis) that [lo, hi] overlaps, clamped to the lattice.
struct Range {
    int lo[3], hi[3];
};
Range Overlap(const Vec3 &lo, const Vec3 &hi, float size, const int count[3])
{
    Range r;
    for (int a = 0; a < 3; a++) {
        // Clamped as floats first so far-away (or huge) coordinates cannot overflow the conversion.
        const float top = static_cast<float>(count[a]);
        r.lo[a] = std::max(0, static_cast<int>(std::floor(std::clamp(lo[a] / size, -1.0f, top))));
        r.hi[a] = std::min(count[a] - 1, static_cast<int>(std::floor(std::clamp(hi[a] / size, -1.0f, top))));
    }
    return r;
}

// The leaf with key (depth, i, j, k), or -1 if that block is not a leaf.
int LeafWithKey(const AdaptiveGrid3D &grid, int depth, int i, int j, int k, float size)
{
    const int leaf = grid.findLeaf(Vec3(i + 0.5f, j + 0.5f, k + 0.5f) * size);
    const AdaptiveGrid3D::Key &key = grid.blocks()[leaf].key;
    return (key.depth == depth && key.i == i && key.j == j && key.k == k) ? leaf : -1;
}

// Refines, one depth at a time, every leaf that a face comes within r0 of. A leaf that a pass does not test
// (one that balance created at or above its depth) is the child of a leaf no face was near, so it is not near
// either, and every leaf left above maxDepth is farther than r0 from the mesh.
void RefineNearFaces(AdaptiveGrid3D &grid, const MeshDistance3D &mesh, float r0, const int blocksPerAxis[3])
{
    const int F = static_cast<int>(mesh.faceCount());
    for (int depth = 0; depth < grid.maxDepth(); depth++) {
        const float size = N * std::ldexp(1.0f, -depth);
        const int count[3] = {blocksPerAxis[0] << depth, blocksPerAxis[1] << depth, blocksPerAxis[2] << depth};
        std::vector<int> marked;
        SPRINGLS_OMP(omp parallel if (F >= kParallelMinHeavyItems))
        {
            std::vector<int> local;
            SPRINGLS_OMP(omp for schedule(dynamic, 256))
            for (int f = 0; f < F; f++) {
                Vec3 lo, hi;
                mesh.faceBounds(f, lo, hi);
                const Range r = Overlap(lo - Vec3::Constant(r0), hi + Vec3::Constant(r0), size, count);
                for (int k = r.lo[2]; k <= r.hi[2]; k++) {
                    for (int j = r.lo[1]; j <= r.hi[1]; j++) {
                        for (int i = r.lo[0]; i <= r.hi[0]; i++) {
                            const Vec3 origin = Vec3(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)) * size;
                            if (!mesh.faceNearBox(f, origin, origin + Vec3::Constant(size), r0)) continue;
                            const int leaf = LeafWithKey(grid, depth, i, j, k, size);
                            if (leaf >= 0) local.push_back(leaf);
                        }
                    }
                }
            }
            SPRINGLS_OMP(omp critical(splat_marks))
            marked.insert(marked.end(), local.begin(), local.end());
        }
        const auto &blocks = grid.blocks();
        std::sort(marked.begin(), marked.end(), [&](int a, int b) { return blocks[a].key < blocks[b].key; });
        marked.erase(std::unique(marked.begin(), marked.end()), marked.end());
        grid.refineLeaves(marked);
    }
}

// Corner values of the maxDepth leaves: exact within `exact` of the mesh, sign * bandWidth beyond (see
// SplatMeshIntoGrid). One chunk (a block region at chunk depth, up to 4^3 maxDepth leaves) at a time: its faces
// (those within `exact` of it) are binned to its maxDepth leaves, so candidate lists only exist for one small
// chunk per thread.
void WriteFineCorners(AdaptiveGrid3D &grid, const MeshDistance3D &mesh, float exact, float bandWidth, const int blocksPerAxis[3], MeshSplatStats &stats)
{
    auto &blocks = grid.blocks();
    const int depth = grid.maxDepth();
    const float size = N * std::ldexp(1.0f, -depth);
    const float halfDiagonal = 0.5f * std::sqrt(3.0f) * size;
    // A corner farther than `exact` from every face has its leaf center's sign when exact > halfDiagonal (the
    // surface would otherwise cross the segment between them, within halfDiagonal of the corner). Otherwise
    // (only for an `exact` below a finest leaf's half diagonal) such corners take the full query instead.
    const bool centerSign = exact > halfDiagonal * (1.0f + 1e-4f);
    const int count[3] = {blocksPerAxis[0] << depth, blocksPerAxis[1] << depth, blocksPerAxis[2] << depth};
    // Chunk depth: two above maxDepth, coarser if the chunk lattice would be large.
    int chunkDepth = std::max(0, depth - 2);
    while (chunkDepth > 0 && static_cast<double>(blocksPerAxis[0]) * blocksPerAxis[1] * blocksPerAxis[2] * std::ldexp(1.0, 3 * chunkDepth) > (1 << 24))
        chunkDepth--;
    const int shift = depth - chunkDepth;
    const int chunkDims[3] = {blocksPerAxis[0] << chunkDepth, blocksPerAxis[1] << chunkDepth, blocksPerAxis[2] << chunkDepth};
    const float chunkSize = N * std::ldexp(1.0f, -chunkDepth);
    const int chunkCount = chunkDims[0] * chunkDims[1] * chunkDims[2];
    const auto chunkOf = [&](int i, int j, int k) { return ((k >> shift) * chunkDims[1] + (j >> shift)) * chunkDims[0] + (i >> shift); };

    // maxDepth leaves per chunk, and each one's slot within its chunk.
    std::vector<int> leafStart(chunkCount + 1, 0), slot(blocks.size(), -1);
    for (int leaf : grid.leaves()) {
        const AdaptiveGrid3D::Key &key = blocks[leaf].key;
        if (key.depth == depth) leafStart[chunkOf(key.i, key.j, key.k) + 1]++;
    }
    std::partial_sum(leafStart.begin(), leafStart.end(), leafStart.begin());
    std::vector<int> leavesOf(leafStart.back());
    {
        std::vector<int> fill(leafStart.begin(), leafStart.end() - 1);
        for (int leaf : grid.leaves()) {
            const AdaptiveGrid3D::Key &key = blocks[leaf].key;
            if (key.depth != depth) continue;
            const int chunk = chunkOf(key.i, key.j, key.k);
            slot[leaf] = fill[chunk] - leafStart[chunk];
            leavesOf[fill[chunk]++] = leaf;
        }
    }
    stats.refinedLeaves = leavesOf.size();

    // Faces per chunk (in increasing order), for the chunks that have maxDepth leaves.
    const int F = static_cast<int>(mesh.faceCount());
    std::vector<int> faceStart(chunkCount + 1, 0);
    std::vector<uint32_t> facesOf;
    for (int pass = 0; pass < 2; pass++) {
        std::vector<int> fill(faceStart.begin(), faceStart.end() - 1);
        for (int f = 0; f < F; f++) {
            Vec3 lo, hi;
            mesh.faceBounds(f, lo, hi);
            const Range r = Overlap(lo - Vec3::Constant(exact), hi + Vec3::Constant(exact), chunkSize, chunkDims);
            for (int k = r.lo[2]; k <= r.hi[2]; k++) {
                for (int j = r.lo[1]; j <= r.hi[1]; j++) {
                    for (int i = r.lo[0]; i <= r.hi[0]; i++) {
                        const int chunk = (k * chunkDims[1] + j) * chunkDims[0] + i;
                        if (leafStart[chunk] == leafStart[chunk + 1]) continue;
                        const Vec3 origin = Vec3(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)) * chunkSize;
                        if (!mesh.faceNearBox(f, origin, origin + Vec3::Constant(chunkSize), exact)) continue;
                        if (pass == 0) {
                            faceStart[chunk + 1]++;
                        } else {
                            facesOf[fill[chunk]++] = static_cast<uint32_t>(f);
                        }
                    }
                }
            }
        }
        if (pass == 0) {
            std::partial_sum(faceStart.begin(), faceStart.end(), faceStart.begin());
            facesOf.resize(faceStart.back());
        }
    }

    std::vector<int> chunks;
    for (int chunk = 0; chunk < chunkCount; chunk++) {
        if (leafStart[chunk] < leafStart[chunk + 1]) chunks.push_back(chunk);
    }
    const int R = static_cast<int>(chunks.size());
    size_t entries = 0, fallback = 0;
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 1) reduction(+ : entries, fallback) if (R >= 2))
    for (int n = 0; n < R; n++) {
        const int chunk = chunks[n];
        const int leafCount = leafStart[chunk + 1] - leafStart[chunk];
        const int *leaves = leavesOf.data() + leafStart[chunk];
        const int ci = chunk % chunkDims[0], cj = (chunk / chunkDims[0]) % chunkDims[1], ck = chunk / (chunkDims[0] * chunkDims[1]);
        const int chunkLo[3] = {ci << shift, cj << shift, ck << shift};
        const int chunkHi[3] = {((ci + 1) << shift) - 1, ((cj + 1) << shift) - 1, ((ck + 1) << shift) - 1};
        // A leaf's corners are within reach[s] = |phi(center)| + half its diagonal of their closest faces, and
        // only faces within `exact` of a corner matter, so each leaf's list holds the faces within
        // min(exact, reach) of its box. A leaf whose center is at least exact + halfDiagonal from the mesh has no
        // corner within `exact`: it gets no list (reach < 0) and all its corners take sign(center) * bandWidth.
        // Slots of the chunk's lattice positions (-1: not a maxDepth leaf).
        const int width = 1 << shift;
        const bool dense = shift <= 5;
        std::vector<int> local(dense ? static_cast<size_t>(width) * width * width : 0, -1);
        std::vector<float> reach(leafCount), centerValue(leafCount);
        for (int s = 0; s < leafCount; s++) {
            const Block &block = blocks[leaves[s]];
            centerValue[s] = mesh.signedDistance(block.origin + Vec3::Constant(0.5f * size), exact + 2.0f * halfDiagonal);
            const float centerDistance = std::abs(centerValue[s]);
            const bool distant = centerDistance - halfDiagonal * (1.0f + 1e-5f) - 1e-5f >= exact;
            reach[s] = distant ? -1.0f : std::min(exact, centerDistance + halfDiagonal * (1.0f + 1e-5f) + 1e-5f);
            if (dense) local[((block.key.k - chunkLo[2]) * width + (block.key.j - chunkLo[1])) * width + (block.key.i - chunkLo[0])] = s;
        }
        // (slot, face) for every face near a leaf, binned by the face's grown bounds.
        std::vector<std::pair<int, uint32_t>> pairs;
        for (int e = faceStart[chunk]; e < faceStart[chunk + 1]; e++) {
            const uint32_t f = facesOf[e];
            Vec3 lo, hi;
            mesh.faceBounds(f, lo, hi);
            Range r = Overlap(lo - Vec3::Constant(exact), hi + Vec3::Constant(exact), size, count);
            for (int a = 0; a < 3; a++) {
                r.lo[a] = std::max(r.lo[a], chunkLo[a]);
                r.hi[a] = std::min(r.hi[a], chunkHi[a]);
            }
            for (int k = r.lo[2]; k <= r.hi[2]; k++) {
                for (int j = r.lo[1]; j <= r.hi[1]; j++) {
                    for (int i = r.lo[0]; i <= r.hi[0]; i++) {
                        int s;
                        if (dense) {
                            s = local[((k - chunkLo[2]) * width + (j - chunkLo[1])) * width + (i - chunkLo[0])];
                        } else {
                            const int leaf = LeafWithKey(grid, depth, i, j, k, size);
                            s = (leaf >= 0) ? slot[leaf] : -1;
                        }
                        if (s < 0 || reach[s] < 0.0f) continue;
                        const Vec3 origin = Vec3(static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)) * size;
                        if (mesh.faceNearBox(f, origin, origin + Vec3::Constant(size), reach[s])) pairs.emplace_back(s, f);
                    }
                }
            }
        }
        // Candidate lists per leaf (counting sort by slot).
        std::vector<int> start(leafCount + 1, 0);
        for (const auto &p : pairs) start[p.first + 1]++;
        std::partial_sum(start.begin(), start.end(), start.begin());
        std::vector<uint32_t> candidates(pairs.size());
        {
            std::vector<int> fill(start.begin(), start.end() - 1);
            for (const auto &p : pairs) candidates[fill[p.first]++] = p.second;
        }
        entries += candidates.size();

        std::vector<std::pair<float, uint32_t>> order;
        std::vector<float> lowerBounds;
        std::vector<std::array<Vec3, 2>> bounds;
        std::vector<uint32_t> sorted, nearby;
        for (int s = 0; s < leafCount; s++) {
            Block &block = blocks[leaves[s]];
            const float outside = (centerValue[s] < 0.0f) ? -bandWidth : bandWidth;
            if (reach[s] < 0.0f) {
                block.makeTile(outside);
                continue;
            }
            // A corner's closest face matters only within `exact`, and it is within U of the corner, the smaller of
            // |phi| at the previous corner plus their distance (distance is 1-Lipschitz) and its distance to the
            // previous corner's closest face. So only faces whose box is within min(U, exact) can be it (ties
            // included): they lie in the prefix of the list sorted by box distance lb from the leaf center with
            // lb <= min(U, exact) + |p - center|, found by scanning local copies of the boxes, and only they go to
            // the exact query. The answer is the same as over the whole list.
            const Vec3 center = block.origin + Vec3::Constant(0.5f * size);
            order.clear();
            for (int e = start[s]; e < start[s + 1]; e++) {
                Vec3 lo, hi;
                mesh.faceBounds(candidates[e], lo, hi);
                order.emplace_back((center - center.cwiseMax(lo).cwiseMin(hi)).norm(), candidates[e]);
            }
            std::sort(order.begin(), order.end());
            const size_t K = order.size();
            sorted.resize(K);
            lowerBounds.resize(K);
            bounds.resize(K);
            for (size_t e = 0; e < K; e++) {
                sorted[e] = order[e].second;
                lowerBounds[e] = order[e].first;
                mesh.faceBounds(sorted[e], bounds[e][0], bounds[e][1]);
            }

            block.materialize();
            float previous = std::numeric_limits<float>::infinity();  // |phi| at the previous corner, if below bandWidth
            Vec3 previousPosition = center;
            int64_t previousFace = -1;  // the previous corner's closest face, if a candidate
            for (int c = 0; c < AdaptiveGrid3D::kBlockNodes; c++) {
                for (int b = 0; b < AdaptiveGrid3D::kBlockNodes; b++) {
                    for (int a = 0; a < AdaptiveGrid3D::kBlockNodes; a++) {
                        const Vec3 p = block.nodePosition(a, b, c);
                        float bound = previous + (p - previousPosition).norm();
                        // On a smooth surface the previous corner's closest face is nearly as close to this one.
                        if (previousFace >= 0) bound = std::min(bound, mesh.faceDistance(static_cast<size_t>(previousFace), p));
                        bound = std::min(bound, exact);
                        // The slack covers float rounding in the bounds.
                        const float within = bound + 1e-4f * (1.0f + bound);
                        const float limit = within + (p - center).norm() * (1.0f + 1e-5f);
                        const size_t prefix = std::upper_bound(lowerBounds.begin(), lowerBounds.end(), limit) - lowerBounds.begin();
                        nearby.clear();
                        const float within2 = within * within;
                        for (size_t e = 0; e < prefix; e++) {
                            if ((p - p.cwiseMax(bounds[e][0]).cwiseMin(bounds[e][1])).squaredNorm() <= within2) nearby.push_back(sorted[e]);
                        }
                        uint32_t closestFace = 0;
                        const std::optional<float> candidate = mesh.signedDistance(p, nearby, exact, &closestFace);
                        float d;
                        previous = std::numeric_limits<float>::infinity();
                        previousFace = -1;
                        if (candidate) {
                            d = *candidate;
                            previous = std::abs(d);
                            previousFace = static_cast<int64_t>(closestFace);
                        } else if (centerSign) {
                            d = outside;
                        } else {
                            d = mesh.signedDistance(p, bandWidth);
                            if (std::abs(d) < bandWidth) previous = std::abs(d);
                            fallback++;
                        }
                        block.phi[Block::index(a, b, c)] = std::clamp(d, -bandWidth, bandWidth);
                        previousPosition = p;
                    }
                }
            }
        }
    }
    stats.candidateEntries = entries;
    stats.fallbackCorners = fallback;
}

// Signs the leaves above maxDepth by face-connected component (see SplatMeshIntoGrid).
void FloodCoarseLeaves(AdaptiveGrid3D &grid, float bandWidth, MeshSplatStats &stats)
{
    auto &blocks = grid.blocks();
    const int depth = grid.maxDepth();
    std::vector<int> coarse, coarseSlot(blocks.size(), -1);
    for (int leaf : grid.leaves()) {
        if (blocks[leaf].key.depth < depth) {
            coarseSlot[leaf] = static_cast<int>(coarse.size());
            coarse.push_back(leaf);
        }
    }
    const int C = static_cast<int>(coarse.size());
    if (C == 0) return;

    // Face neighbors (four probes per face, as in the grid's own neighbor slots), border votes, and whether a
    // leaf touches the domain boundary.
    constexpr int kProbes = 6 * 4;
    std::vector<std::array<int, kProbes>> neighbors(C);
    std::vector<long long> positive(C, 0), negative(C, 0);
    std::vector<uint8_t> boundary(C, 0);
    const Vec3 domainMax = grid.domainMax();
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 16) if (C >= kParallelMinBlocks))
    for (int n = 0; n < C; n++) {
        const Block &block = blocks[coarse[n]];
        const float size = block.size(), eps = 0.25f * block.h;
        for (int face = 0; face < 6; face++) {
            const int axis = face / 2, side = face % 2;
            int previous = -1;
            for (int q = 0; q < 4; q++) {
                int &neighbor = neighbors[n][face * 4 + q];
                neighbor = -1;
                Vec3 p = block.origin;
                p[(axis + 1) % 3] += size * ((q & 1) ? 0.75f : 0.25f);
                p[(axis + 2) % 3] += size * ((q & 2) ? 0.75f : 0.25f);
                p[axis] += side ? size + eps : -eps;
                if (p[axis] < 0.0f || p[axis] > domainMax[axis]) {
                    boundary[n] = 1;
                    continue;
                }
                neighbor = grid.findLeaf(p);
                if (neighbor == previous) continue;  // a neighbor at least as coarse spans the face
                previous = neighbor;
                const Block &other = blocks[neighbor];
                if (other.key.depth != depth) continue;
                // The maxDepth neighbor's face against this leaf (exact values; it is finer, so wholly on it).
                for (int u = 0; u < AdaptiveGrid3D::kBlockNodes; u++) {
                    for (int v = 0; v < AdaptiveGrid3D::kBlockNodes; v++) {
                        int node[3];
                        node[axis] = side ? 0 : N;
                        node[(axis + 1) % 3] = u;
                        node[(axis + 2) % 3] = v;
                        const float value = other.at(node[0], node[1], node[2]);
                        if (value > 0.0f) positive[n]++;
                        if (value < 0.0f) negative[n]++;
                    }
                }
            }
        }
    }

    // Components (union-find), numbered in leaf order.
    std::vector<int> parent(C);
    std::iota(parent.begin(), parent.end(), 0);
    const auto find = [&](int x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    for (int n = 0; n < C; n++) {
        for (int neighbor : neighbors[n]) {
            if (neighbor < 0 || coarseSlot[neighbor] < 0) continue;
            const int a = find(n), b = find(coarseSlot[neighbor]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
    }
    std::vector<int> component(C, -1), rootComponent(C, -1);
    std::vector<long long> votes;  // positive minus negative border corners
    std::vector<uint8_t> touches, voted;
    for (int n = 0; n < C; n++) {
        const int root = find(n);
        if (rootComponent[root] < 0) {
            rootComponent[root] = static_cast<int>(votes.size());
            votes.push_back(0);
            touches.push_back(0);
            voted.push_back(0);
        }
        const int c = component[n] = rootComponent[root];
        votes[c] += positive[n] - negative[n];
        touches[c] |= boundary[n];
        voted[c] |= (positive[n] + negative[n]) > 0;
    }
    stats.floodComponents = votes.size();

    SPRINGLS_OMP(omp parallel for if (C >= kParallelMinBlocks))
    for (int n = 0; n < C; n++) {
        const int c = component[n];
        const bool outside = voted[c] ? votes[c] >= 0 : touches[c] != 0;
        blocks[coarse[n]].makeTile(outside ? bandWidth : -bandWidth);
    }
}
}  // namespace

MeshSplatStats SplatMeshIntoGrid(AdaptiveGrid3D &grid, const MeshDistance3D &mesh, float r0, float bandWidth)
{
    ProfileScope profileScope("SplatMeshIntoGrid");
    MeshSplatStats stats;
    if (grid.leaves().empty()) return stats;
    const Vec3 domainMax = grid.domainMax();
    const int blocksPerAxis[3] = {static_cast<int>(std::lround(domainMax.x() / N)), static_cast<int>(std::lround(domainMax.y() / N)),
                                  static_cast<int>(std::lround(domainMax.z() / N))};
    std::optional<ProfileScope> phase(std::in_place, "SplatMeshIntoGrid: refine");
    RefineNearFaces(grid, mesh, r0, blocksPerAxis);
    phase.emplace("SplatMeshIntoGrid: corners");
    // Exact within r0, but never short of the active band of a leaf one depth coarser than the finest (the
    // coarsest that shares corners with finest leaves), so the active set is the one exact values everywhere
    // would give. (r0 from initLevelSet() covers it only up to about depth 5, hence the max().)
    const float exact = std::max(r0, grid.activeThreshold(std::ldexp(1.0f, 1 - grid.maxDepth())));
    WriteFineCorners(grid, mesh, exact, bandWidth, blocksPerAxis, stats);
    phase.emplace("SplatMeshIntoGrid: flood");
    FloodCoarseLeaves(grid, bandWidth, stats);
    phase.reset();
    grid.synchronize();
    grid.updateActive();
    return stats;
}
}  // namespace springls
