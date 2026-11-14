#include <springls/AdaptiveGrid2D.hpp>

#include <springls/Contour2D.hpp>

#include <cassert>

#include <algorithm>
#include <cmath>
#include <limits>

namespace springls
{
namespace
{
// Pseudo-time step of evolve() in cell units (matches SpringLevelSet2D::updateSignedLevelSet).
constexpr float kEvolveTimeStep = 0.5f;
// Velocity magnitudes are clamped by the caller; curvature motion is clamped here (cell units).
constexpr float kMaxCurvatureForce = 10.0f;
// Crossings are welded when they agree to within a few float ulps of the domain size: enough to
// merge the same crossing computed by a coarse and a fine leaf, far below any real separation.
constexpr double kWeldUlps = 4.0;
// Corner values closer to zero than this many weld tolerances are pushed to +/- that value, so a
// crossing never lands on (or within weld distance of) a corner, where crossings on the two edges
// meeting at that corner would weld into one vertex and break the curve.
constexpr double kZeroEpsilonWelds = 50.0;
// Dangling curve ends (a nudge can move a coarse/fine crossing by up to the zero epsilon) are
// joined to the nearest dangling start within this many zero epsilons.
constexpr double kRepairEpsilons = 4.0;
// A tile is re-materialized when a shared corner differs from its value by more than this.
constexpr float kTileTolerance = 1E-6f;

using Block = AdaptiveGrid2D::Block;

// Bilinear interpolation of a block at local (cell-unit) coordinates (u, v).
float Bilinear(const Block &block, float u, float v)
{
    if (block.isTile()) return block.tileValue;
    const int a = std::clamp(static_cast<int>(std::floor(u)), 0, AdaptiveGrid2D::kBlockCells - 1);
    const int b = std::clamp(static_cast<int>(std::floor(v)), 0, AdaptiveGrid2D::kBlockCells - 1);
    const float fx = std::clamp(u - a, 0.0f, 1.0f);
    const float fy = std::clamp(v - b, 0.0f, 1.0f);
    const auto at = [&](int i, int j) { return block.phi[Block::index(i, j)]; };
    return (1 - fy) * ((1 - fx) * at(a, b) + fx * at(a + 1, b)) + fy * ((1 - fx) * at(a, b + 1) + fx * at(a + 1, b + 1));
}
}  // namespace

void AdaptiveGrid2D::clear()
{
    mBlocks.clear();
    mIndex.clear();
    mFreeList.clear();
    mLeaves.clear();
    mActiveLeaves.clear();
    mSyncGroupStart.clear();
    mSyncEntries.clear();
    mMidpointRules.clear();
    mPlansValid = false;
    mBlocksX = mBlocksY = 0;
}

void AdaptiveGrid2D::initialize(const ImageF &levelSet, int maxDepth, float bandWidth)
{
    clear();
    mMaxDepth = std::clamp(maxDepth, 0, kMaxSupportedDepth);
    mBandWidth = bandWidth;
    const int W = levelSet.width();
    const int H = levelSet.height();
    mBlocksX = std::max(1, (W - 1 + kBlockCells - 1) / kBlockCells);
    mBlocksY = std::max(1, (H - 1 + kBlockCells - 1) / kBlockCells);
    mDomainMax = Vec2(static_cast<float>(mBlocksX * kBlockCells), static_cast<float>(mBlocksY * kBlockCells));
    for (int bj = 0; bj < mBlocksY; bj++) {
        for (int bi = 0; bi < mBlocksX; bi++) {
            Block block;
            block.key = Key{0, bi, bj};
            block.h = 1.0f;
            block.origin = Vec2(static_cast<float>(bi * kBlockCells), static_cast<float>(bj * kBlockCells));
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int x = std::min(bi * kBlockCells + a, W - 1);
                    const int y = std::min(bj * kBlockCells + b, H - 1);
                    block.at(a, b) = std::clamp(levelSet.at(x, y), -mBandWidth, mBandWidth);
                }
            }
            allocate(block);
        }
    }
    rebuildLeafList();
    updateActive();
    compactTiles();
}

int AdaptiveGrid2D::allocate(const Block &block)
{
    int index;
    if (!mFreeList.empty()) {
        index = mFreeList.back();
        mFreeList.pop_back();
        mBlocks[index] = block;
    } else {
        index = static_cast<int>(mBlocks.size());
        mBlocks.push_back(block);
    }
    mBlocks[index].alive = true;
    mIndex[block.key] = index;
    return index;
}

void AdaptiveGrid2D::release(int index)
{
    mIndex.erase(mBlocks[index].key);
    mBlocks[index].alive = false;
    mBlocks[index].makeTile(0.0f);  // free its array
    mFreeList.push_back(index);
}

void AdaptiveGrid2D::rebuildLeafList()
{
    invalidatePlans();
    mLeaves.clear();
    for (int n = 0; n < static_cast<int>(mBlocks.size()); n++) {
        if (mBlocks[n].alive && mBlocks[n].isLeaf()) mLeaves.push_back(n);
    }
    std::sort(mLeaves.begin(), mLeaves.end(), [this](int a, int b) { return mBlocks[a].key < mBlocks[b].key; });
}

int AdaptiveGrid2D::findLeaf(const Vec2 &q) const
{
    const Vec2 p(std::clamp(q.x(), 0.0f, mDomainMax.x()), std::clamp(q.y(), 0.0f, mDomainMax.y()));
    const int bi = std::clamp(static_cast<int>(std::floor(p.x() / kBlockCells)), 0, mBlocksX - 1);
    const int bj = std::clamp(static_cast<int>(std::floor(p.y() / kBlockCells)), 0, mBlocksY - 1);
    int index = mIndex.at(Key{0, bi, bj});
    while (!mBlocks[index].isLeaf()) {
        const Block &block = mBlocks[index];
        const float half = 0.5f * block.size();
        const int cx = (p.x() >= block.origin.x() + half) ? 1 : 0;
        const int cy = (p.y() >= block.origin.y() + half) ? 1 : 0;
        index = block.children[cx + 2 * cy];
    }
    return index;
}

float AdaptiveGrid2D::sample(const Vec2 &q) const
{
    const Vec2 p(std::clamp(q.x(), 0.0f, mDomainMax.x()), std::clamp(q.y(), 0.0f, mDomainMax.y()));
    const Block &block = mBlocks[findLeaf(p)];
    const Vec2 u = (p - block.origin) / block.h;
    return Bilinear(block, u.x(), u.y());
}

bool AdaptiveGrid2D::refine(int index)
{
    if (!mBlocks[index].isLeaf()) return true;
    const int depth = mBlocks[index].key.depth;
    if (depth >= mMaxDepth) return false;
    // 2:1 balance: every leaf touching this block must be at least as fine before it splits.
    {
        const Vec2 center = mBlocks[index].origin + Vec2::Constant(0.5f * mBlocks[index].size());
        const float reach = 0.5f * mBlocks[index].size() + 0.5f * mBlocks[index].h;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                const Vec2 p = center + reach * Vec2(static_cast<float>(dx), static_cast<float>(dy));
                if (p.x() < 0 || p.y() < 0 || p.x() > mDomainMax.x() || p.y() > mDomainMax.y()) continue;
                int neighbor = findLeaf(p);
                while (mBlocks[neighbor].key.depth < depth) {
                    refine(neighbor);
                    neighbor = findLeaf(p);
                }
            }
        }
    }
    const Block parent = mBlocks[index];  // copy: allocate() may reallocate mBlocks
    for (int c = 0; c < 4; c++) {
        const int cx = c & 1;
        const int cy = c >> 1;
        Block child;
        child.key = Key{depth + 1, 2 * parent.key.i + cx, 2 * parent.key.j + cy};
        child.h = 0.5f * parent.h;
        child.origin = parent.origin + 0.5f * parent.size() * Vec2(static_cast<float>(cx), static_cast<float>(cy));
        child.parent = index;
        child.active = parent.active;
        if (parent.isTile()) {
            child.makeTile(parent.tileValue);
        } else {
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const Vec2 u = (child.nodePosition(a, b) - parent.origin) / parent.h;
                    child.at(a, b) = Bilinear(parent, u.x(), u.y());
                }
            }
        }
        const int childIndex = allocate(child);
        mBlocks[index].children[c] = childIndex;
    }
    mBlocks[index].makeTile(0.0f);  // interior blocks keep no values (leaves are authoritative)
    rebuildLeafList();
    return true;
}

bool AdaptiveGrid2D::canCoarsen(int index) const
{
    const Block &block = mBlocks[index];
    if (!block.alive || block.isLeaf()) return false;
    for (int c : block.children) {
        if (!mBlocks[c].isLeaf()) return false;
    }
    // Every leaf touching the block must be at most one depth finer than the merged block.
    constexpr int kProbesPerSide = 8;
    const float size = block.size();
    const float step = size / kProbesPerSide;
    const float eps = 0.125f * step;
    for (int s = 0; s <= kProbesPerSide; s++) {
        const float t = std::min(s * step + 0.5f * step, size - 0.5f * step);
        const Vec2 probes[4] = {block.origin + Vec2(-eps, t), block.origin + Vec2(size + eps, t), block.origin + Vec2(t, -eps),
                                block.origin + Vec2(t, size + eps)};
        for (const Vec2 &p : probes) {
            if (p.x() < 0 || p.y() < 0 || p.x() > mDomainMax.x() || p.y() > mDomainMax.y()) continue;
            if (mBlocks[findLeaf(p)].key.depth > block.key.depth + 1) return false;
        }
    }
    const Vec2 corners[4] = {block.origin + Vec2(-eps, -eps), block.origin + Vec2(size + eps, -eps), block.origin + Vec2(-eps, size + eps),
                             block.origin + Vec2(size + eps, size + eps)};
    for (const Vec2 &p : corners) {
        if (p.x() < 0 || p.y() < 0 || p.x() > mDomainMax.x() || p.y() > mDomainMax.y()) continue;
        if (mBlocks[findLeaf(p)].key.depth > block.key.depth + 1) return false;
    }
    return true;
}

void AdaptiveGrid2D::coarsen(int index)
{
    if (!canCoarsen(index)) return;
    // Restriction by injection: every coarse corner is a corner of one of the children.
    const std::array<int, 4> children = mBlocks[index].children;
    bool allTiles = true;
    for (int c : children) allTiles &= mBlocks[c].isTile() && mBlocks[c].tileValue == mBlocks[children[0]].tileValue;
    if (allTiles) {
        mBlocks[index].makeTile(mBlocks[children[0]].tileValue);
    } else {
        std::vector<float> values(kNodeCount);
        for (int b = 0; b < kBlockNodes; b++) {
            for (int a = 0; a < kBlockNodes; a++) values[Block::index(a, b)] = sample(mBlocks[index].nodePosition(a, b));
        }
        mBlocks[index].phi = std::move(values);
    }
    bool anyActive = false;
    for (int c : children) {
        anyActive |= mBlocks[c].active;
        release(c);
    }
    mBlocks[index].children = {-1, -1, -1, -1};
    mBlocks[index].active = anyActive;
    rebuildLeafList();
}

void AdaptiveGrid2D::buildPlans()
{
    mSyncGroupStart.clear();
    mSyncEntries.clear();
    mMidpointRules.clear();
    const double finestScale = std::ldexp(1.0, mMaxDepth);
    const auto latticeKey = [&](const Vec2 &p) {
        const int64_t X = std::llround(p.x() * finestScale);
        const int64_t Y = std::llround(p.y() * finestScale);
        return (X << 32) ^ (Y & 0xffffffff);
    };
    const auto inDomain = [&](const Vec2 &p) { return p.x() >= 0 && p.y() >= 0 && p.x() <= mDomainMax.x() && p.y() <= mDomainMax.y(); };
    // Shared corners: every leaf corner position on a block border, with all leaves having it as a corner.
    std::unordered_map<int64_t, bool> seenPosition;
    for (int index : mLeaves) {
        const Block &block = mBlocks[index];
        const float delta = 0.125f * block.h;
        for (int b = 0; b < kBlockNodes; b++) {
            for (int a = 0; a < kBlockNodes; a++) {
                if (a != 0 && b != 0 && a != kBlockCells && b != kBlockCells) continue;
                const Vec2 p = block.nodePosition(a, b);
                if (!seenPosition.emplace(latticeKey(p), true).second) continue;
                const size_t start = mSyncEntries.size();
                int seen[4] = {-1, -1, -1, -1};
                for (int k = 0; k < 4; k++) {
                    const Vec2 probe = p + Vec2((k & 1) ? delta : -delta, (k & 2) ? delta : -delta);
                    if (!inDomain(probe)) continue;
                    const int leaf = findLeaf(probe);
                    if (std::find(std::begin(seen), std::end(seen), leaf) != std::end(seen)) continue;
                    seen[k] = leaf;
                    const Block &other = mBlocks[leaf];
                    const Vec2 u = (p - other.origin) / other.h;
                    const int ua = static_cast<int>(std::lround(u.x()));
                    const int ub = static_cast<int>(std::lround(u.y()));
                    if (std::abs(u.x() - ua) > 1E-3f || std::abs(u.y() - ub) > 1E-3f) continue;  // not a corner of `other`
                    if (ua < 0 || ub < 0 || ua > kBlockCells || ub > kBlockCells) continue;
                    mSyncEntries.push_back({leaf, static_cast<uint8_t>(Block::index(ua, ub)), static_cast<uint8_t>(other.key.depth)});
                }
                if (mSyncEntries.size() - start > 1) {
                    mSyncGroupStart.push_back(static_cast<int>(start));
                } else {
                    mSyncEntries.resize(start);  // a corner only one leaf owns needs no synchronization
                }
            }
        }
    }
    mSyncGroupStart.push_back(static_cast<int>(mSyncEntries.size()));
    mSyncEntries.shrink_to_fit();
    // Neighbor leaves (two per side, one per corner) and coarse/fine midpoint rules.
    for (int index : mLeaves) {
        Block &block = mBlocks[index];
        const float size = block.size();
        const float eps = 0.25f * block.h;
        const Vec2 o = block.origin;
        const Vec2 probes[kNeighborSlots] = {o + Vec2(-eps, 0.25f * size),
                                             o + Vec2(-eps, 0.75f * size),  // -x side
                                             o + Vec2(size + eps, 0.25f * size),
                                             o + Vec2(size + eps, 0.75f * size),  // +x side
                                             o + Vec2(0.25f * size, -eps),
                                             o + Vec2(0.75f * size, -eps),  // -y side
                                             o + Vec2(0.25f * size, size + eps),
                                             o + Vec2(0.75f * size, size + eps),  // +y side
                                             o + Vec2(-eps, -eps),
                                             o + Vec2(size + eps, -eps),
                                             o + Vec2(-eps, size + eps),
                                             o + Vec2(size + eps, size + eps)};
        for (int s = 0; s < kNeighborSlots; s++) {
            block.neighbors[s] = inDomain(probes[s]) ? findLeaf(probes[s]) : -1;
        }
        for (int side = 0; side < 4; side++) {
            const int neighbor = block.neighbors[2 * side];
            if (neighbor < 0 || mBlocks[neighbor].key.depth >= block.key.depth) continue;
            for (int t = 1; t < kBlockCells; t += 2) {
                if (side < 2) {
                    const int a = (side == 0) ? 0 : kBlockCells;
                    mMidpointRules.push_back({index, static_cast<uint8_t>(Block::index(a, t)), static_cast<uint8_t>(Block::index(a, t - 1)),
                                              static_cast<uint8_t>(Block::index(a, t + 1))});
                } else {
                    const int b = (side == 2) ? 0 : kBlockCells;
                    mMidpointRules.push_back({index, static_cast<uint8_t>(Block::index(t, b)), static_cast<uint8_t>(Block::index(t - 1, b)),
                                              static_cast<uint8_t>(Block::index(t + 1, b))});
                }
            }
        }
    }
    mMidpointRules.shrink_to_fit();
    mPlansValid = true;
}

void AdaptiveGrid2D::synchronize()
{
    if (!mPlansValid) buildPlans();
    // Shared corners take the average of the finest leaves that have them.
    for (size_t g = 0; g + 1 < mSyncGroupStart.size(); g++) {
        const int begin = mSyncGroupStart[g];
        const int end = mSyncGroupStart[g + 1];
        int depth = -1;
        for (int e = begin; e < end; e++) depth = std::max(depth, static_cast<int>(mSyncEntries[e].depth));
        double sum = 0.0;
        int count = 0;
        for (int e = begin; e < end; e++) {
            if (mSyncEntries[e].depth == depth) {
                sum += mBlocks[mSyncEntries[e].block].value(mSyncEntries[e].node);
                count++;
            }
        }
        const float value = static_cast<float>(sum / count);
        for (int e = begin; e < end; e++) {
            Block &block = mBlocks[mSyncEntries[e].block];
            if (block.isTile() && std::abs(block.tileValue - value) <= kTileTolerance) continue;
            block.materialize();
            block.phi[mSyncEntries[e].node] = value;
        }
    }
    // Along a coarse/fine border the fine leaf's in-between corners interpolate linearly.
    for (const MidpointRule &rule : mMidpointRules) {
        Block &block = mBlocks[rule.block];
        if (block.isTile()) continue;  // a tile is linear (constant) already
        block.phi[rule.node] = 0.5f * (block.phi[rule.nodeA] + block.phi[rule.nodeB]);
    }
}

void AdaptiveGrid2D::gatherLocal(int index, std::array<float, kGhostWidth * kGhostWidth> &local) const
{
    const Block &block = mBlocks[index];
    const auto slotFor = [&](int a, int b) {
        // Corners first, then sides (lower / upper half of the side).
        if (a < 0 && b < 0) return 8;
        if (a > kBlockCells && b < 0) return 9;
        if (a < 0 && b > kBlockCells) return 10;
        if (a > kBlockCells && b > kBlockCells) return 11;
        if (a < 0) return (b < kBlockCells / 2) ? 0 : 1;
        if (a > kBlockCells) return (b < kBlockCells / 2) ? 2 : 3;
        if (b < 0) return (a < kBlockCells / 2) ? 4 : 5;
        return (a < kBlockCells / 2) ? 6 : 7;
    };
    for (int b = -1; b <= kBlockNodes; b++) {
        for (int a = -1; a <= kBlockNodes; a++) {
            float &out = local[(b + 1) * kGhostWidth + (a + 1)];
            if (a >= 0 && b >= 0 && a < kBlockNodes && b < kBlockNodes) {
                out = block.at(a, b);
                continue;
            }
            const int neighbor = block.neighbors[slotFor(a, b)];
            if (neighbor < 0) {
                // Outside the domain: replicate the border corner.
                out = block.at(std::clamp(a, 0, kBlockCells), std::clamp(b, 0, kBlockCells));
                continue;
            }
            const Block &src = mBlocks[neighbor];
            const Vec2 u = (block.nodePosition(a, b) - src.origin) / src.h;
            out = Bilinear(src, u.x(), u.y());
        }
    }
}

void AdaptiveGrid2D::updateActive()
{
    const float threshold = mBandWidth * (1.0f - 1E-4f);
    mActiveLeaves.clear();
    for (int index : mLeaves) {
        Block &block = mBlocks[index];
        float minAbs = std::abs(block.tileValue);
        if (!block.isTile()) {
            minAbs = std::numeric_limits<float>::max();
            for (float v : block.phi) minAbs = std::min(minAbs, std::abs(v));
        }
        block.active = minAbs < threshold;
        if (block.active) mActiveLeaves.push_back(index);
    }
}

void AdaptiveGrid2D::compactTiles()
{
    for (int index : mLeaves) {
        Block &block = mBlocks[index];
        if (block.isTile() || block.active) continue;
        const float first = block.phi[0];
        if (std::abs(first) < mBandWidth) continue;
        bool uniform = true;
        for (float v : block.phi) {
            if (v != first) {
                uniform = false;
                break;
            }
        }
        if (uniform) block.makeTile(first);
    }
}

void AdaptiveGrid2D::evolve(int iterations, float curvatureWeight, const std::vector<NodeVelocities> &velocities)
{
    if (!mPlansValid) buildPlans();
    const std::vector<int> &active = mActiveLeaves;
    assert(velocities.size() == active.size());
    std::vector<std::array<float, kNodeCount>> next(active.size());
    constexpr int kLayers = 3;
    constexpr int kStride[2] = {1, kGhostWidth};
    for (int it = 0; it < iterations; it++) {
        SPRINGLS_OMP(omp parallel for if(active.size() >= kParallelMinBlocks))
        for (int n = 0; n < static_cast<int>(active.size()); n++) {
            const int index = active[n];
            std::array<float, kGhostWidth * kGhostWidth> local;
            gatherLocal(index, local);
            const Block &block = mBlocks[index];
            const float h = block.h, invH = 1.0f / h;
            const auto L = [](int a, int b) { return (b + 1) * kGhostWidth + (a + 1); };
            // 1. The zero layer (|phi| <= h / 2) moves; everything else is rebuilt from it below
            //    (the sparse-field method, as NarrowBand2D does on the uniform grid).
            std::array<float, kGhostWidth * kGhostWidth> updated = local;
            std::array<uint8_t, kNodeCount> zeroLayer{};
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int l = L(a, b);
                    if (std::abs(local[l]) > 0.5f * h) continue;
                    zeroLayer[Block::index(a, b)] = 1;
                    // Same upwind scheme as SpringLevelSet2D::distanceFieldSpeed, in this block's cell units.
                    const auto v = [&](int da, int db) { return local[l + da + db * kGhostWidth] * invH; };
                    const float v11 = v(0, 0);
                    const float v01 = v(-1, 0), v10 = v(0, -1), v21 = v(1, 0), v12 = v(0, 1);
                    const float DxNeg = v11 - v01, DxPos = v21 - v11;
                    const float DyNeg = v11 - v10, DyPos = v12 - v11;
                    float kappa = 0.0f;
                    if (curvatureWeight != 0.0f) {
                        const float v00 = v(-1, -1), v20 = v(1, -1), v22 = v(1, 1), v02 = v(-1, 1);
                        const float DxCtr = 0.5f * (v21 - v01), DyCtr = 0.5f * (v12 - v10);
                        const float DxxCtr = v21 - v11 - v11 + v01, DyyCtr = v12 - v11 - v11 + v10;
                        const float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;
                        const float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
                        const float denom = DxCtr * DxCtr + DyCtr * DyCtr;
                        kappa = (std::abs(denom) > 1E-5f) ? curvatureWeight * numer / denom : 0.0f;
                        kappa = std::clamp(kappa, -kMaxCurvatureForce, kMaxCurvatureForce);
                    }
                    const Vec2 g = velocities[n][Block::index(a, b)];
                    float advection = 0.0f;
                    if (g.x() > 0) advection = g.x() * DxNeg;
                    else if (g.x() < 0) advection = g.x() * DxPos;
                    if (g.y() > 0) advection += g.y() * DyNeg;
                    else if (g.y() < 0) advection += g.y() * DyPos;
                    // At most half a cell per iteration, like the uniform narrow band.
                    updated[l] = (v11 + kEvolveTimeStep * std::clamp(kappa - advection, -1.0f, 1.0f)) * h;
                }
            }
            // 2. Distance layers 1..kLayers around the zero layer (in place; neighbor leaves' layers
            //    are a sweep behind across block borders until the next iteration).
            for (int band = 1; band <= kLayers; band++) {
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int a = 0; a < kBlockNodes; a++) {
                        if (zeroLayer[Block::index(a, b)]) continue;
                        const int l = L(a, b);
                        const float old = updated[l];
                        float v = old;
                        if (v < (-band + 0.5f) * h) {
                            v = -mBandWidth;
                            for (int ax = 0; ax < 2; ax++) {
                                for (int sgn = -1; sgn <= 1; sgn += 2) {
                                    const float u = updated[l + sgn * kStride[ax]];
                                    if (u <= h) v = std::max(v, u);
                                }
                            }
                            v -= h;
                        } else if (v > (band - 0.5f) * h) {
                            v = mBandWidth;
                            for (int ax = 0; ax < 2; ax++) {
                                for (int sgn = -1; sgn <= 1; sgn += 2) {
                                    const float u = updated[l + sgn * kStride[ax]];
                                    if (u >= -h) v = std::min(v, u);
                                }
                            }
                            v += h;
                        }
                        updated[l] = (old * v > 0.0f) ? v : old;
                    }
                }
            }
            // 3. Plug: a zero-layer node whose four neighbors all lie on its side is not on the interface.
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int l = L(a, b);
                    const float v = updated[l];
                    const float sgn = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
                    if (sgn == 0.0f || std::abs(v) >= 0.5f * h) continue;
                    bool plugged = true;
                    for (int ax = 0; ax < 2 && plugged; ax++) {
                        for (int k = -1; k <= 1; k += 2) plugged &= sgn * updated[l + k * kStride[ax]] > 0.0f;
                    }
                    if (plugged) updated[l] = sgn * h;
                }
            }
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) next[n][Block::index(a, b)] = std::clamp(updated[L(a, b)], -mBandWidth, mBandWidth);
            }
        }
        for (size_t n = 0; n < active.size(); n++) {
            Block &block = mBlocks[active[n]];
            block.phi.assign(next[n].begin(), next[n].end());
        }
        synchronize();
    }
}

int AdaptiveGrid2D::correctSigns(const std::function<bool(const Vec2 &, float, float &)> &measure)
{
    const std::vector<int> &active = mActiveLeaves;
    const int A = static_cast<int>(active.size());
    int corrected = 0;
    SPRINGLS_OMP(omp parallel for reduction(+ : corrected) if (A >= kParallelMinBlocks))
    for (int n = 0; n < A; n++) {
        Block &block = mBlocks[active[n]];
        if (block.isTile()) continue;
        for (int b = 0; b < kBlockNodes; b++) {
            for (int a = 0; a < kBlockNodes; a++) {
                float d;
                if (!measure(block.nodePosition(a, b), block.h, d)) continue;
                float &v = block.phi[Block::index(a, b)];
                if ((d < 0.0f) == (v < 0.0f)) continue;
                v = std::clamp(d, -mBandWidth, mBandWidth);
                corrected++;
            }
        }
    }
    // Shared corners are measured identically by every leaf that has them (same position).
    if (corrected > 0) synchronize();
    return corrected;
}

void AdaptiveGrid2D::extractContour(Points2 &vertices, Curves &curves, std::vector<float> &vertexScales) const
{
    vertices.clear();
    curves.clear();
    vertexScales.clear();
    const double weld = kWeldUlps * std::max<double>({mDomainMax.x(), mDomainMax.y(), 1.0}) * std::numeric_limits<float>::epsilon();
    const float zeroEpsilon = static_cast<float>(kZeroEpsilonWelds * weld);
    std::unordered_map<int64_t, uint32_t> welded;
    const auto vertexOf = [&](const Vec2 &p, float scale) {
        const int64_t X = std::llround(p.x() / weld);
        const int64_t Y = std::llround(p.y() / weld);
        const int64_t key = (X << 32) ^ (Y & 0xffffffff);
        const auto it = welded.find(key);
        if (it != welded.end()) {
            vertexScales[it->second] = std::min(vertexScales[it->second], scale);
            return it->second;
        }
        const uint32_t id = static_cast<uint32_t>(vertices.size());
        vertices.push_back(p);
        vertexScales.push_back(scale);
        welded.emplace(key, id);
        return id;
    };
    std::vector<std::pair<uint32_t, uint32_t>> segments;  // oriented: inside (phi < 0) on the right
    for (int index : mLeaves) {
        const Block &block = mBlocks[index];
        if (block.isTile()) continue;  // a tile has one sign everywhere
        for (int b = 0; b < kBlockCells; b++) {
            for (int a = 0; a < kBlockCells; a++) {
                // Corners counter-clockwise: c0 (a,b), c1 (a+1,b), c2 (a+1,b+1), c3 (a,b+1).
                const int ca[4] = {a, a + 1, a + 1, a};
                const int cb[4] = {b, b, b + 1, b + 1};
                float val[4];
                Vec2 pos[4];
                for (int k = 0; k < 4; k++) {
                    val[k] = block.at(ca[k], cb[k]);
                    // Push values away from zero (zero itself counts as outside).
                    if (std::abs(val[k]) < zeroEpsilon) val[k] = (val[k] < 0.0f) ? -zeroEpsilon : zeroEpsilon;
                    pos[k] = block.nodePosition(ca[k], cb[k]);
                }
                Vec2 cross[4];
                MarchSquare(val, pos, cross, [&](int e0, int e1) {
                    const uint32_t u = vertexOf(cross[e0], block.h);
                    const uint32_t v = vertexOf(cross[e1], block.h);
                    if (u != v) segments.emplace_back(u, v);
                });
            }
        }
    }
    // Chain oriented segments into curves.
    const size_t V = vertices.size();
    std::vector<int> nextOf, inDegree;
    LinkSegments(V, segments, nextOf, inDegree);
    // Repair: join each dangling end (no successor) away from the domain border to the nearest
    // dangling start (no predecessor) within the repair distance, or within one cell of the coarser
    // of the two vertexes, whichever is larger. A nudge moves a crossing along its edge by
    // zeroEpsilon / (slope of phi along the edge), so by far more than the zero epsilon where the
    // contour runs almost parallel to the edge; but the nudged crossing stays on its edge, so a fine
    // crossing and the coarse crossing of the same border point stay within one coarse cell of each
    // other. Ends only dangle at such seams, so the cell-sized reach cannot join two parts of a
    // correct contour.
    {
        const float repair = static_cast<float>(kRepairEpsilons) * zeroEpsilon;
        const auto onBorder = [&](const Vec2 &p) {
            return p.x() <= repair || p.y() <= repair || p.x() >= mDomainMax.x() - repair || p.y() >= mDomainMax.y() - repair;
        };
        std::vector<uint32_t> ends, starts;
        for (uint32_t v = 0; v < V; v++) {
            if (nextOf[v] < 0 && inDegree[v] > 0 && !onBorder(vertices[v])) ends.push_back(v);
            if (inDegree[v] == 0 && nextOf[v] >= 0 && !onBorder(vertices[v])) starts.push_back(v);
        }
        std::vector<bool> used(starts.size(), false);
        for (uint32_t e : ends) {
            int best = -1;
            float bestDistance = std::numeric_limits<float>::infinity();
            for (size_t k = 0; k < starts.size(); k++) {
                const float d = distance(vertices[e], vertices[starts[k]]);
                const float reach = std::max(repair, std::max(vertexScales[e], vertexScales[starts[k]]));
                if (!used[k] && starts[k] != e && d <= reach && d <= bestDistance) {
                    best = static_cast<int>(k);
                    bestDistance = d;
                }
            }
            if (best >= 0) {
                nextOf[e] = static_cast<int>(starts[best]);
                inDegree[starts[best]]++;
                used[best] = true;
            }
        }
    }
    WalkCurves(nextOf, inDegree, curves);
}

size_t AdaptiveGrid2D::tileCount() const
{
    size_t tiles = 0;
    for (int index : mLeaves) tiles += mBlocks[index].isTile() ? 1 : 0;
    return tiles;
}

size_t AdaptiveGrid2D::memoryBytes() const
{
    size_t bytes = sizeof(*this);
    bytes += mBlocks.capacity() * sizeof(Block);
    for (const Block &block : mBlocks) bytes += block.phi.capacity() * sizeof(float);
    // unordered_map: one node (key, value, next pointer, cached hash) plus one bucket per entry.
    bytes += mIndex.size() * (sizeof(Key) + sizeof(int) + 2 * sizeof(void *)) + mIndex.bucket_count() * sizeof(void *);
    bytes += (mFreeList.capacity() + mLeaves.capacity() + mActiveLeaves.capacity() + mSyncGroupStart.capacity()) * sizeof(int);
    bytes += mSyncEntries.capacity() * sizeof(SyncEntry) + mMidpointRules.capacity() * sizeof(MidpointRule);
    return bytes;
}
}  // namespace springls
