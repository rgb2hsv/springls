#include <spatial/AdaptiveGrid2D.hpp>

#include <levelset/Contour2D.hpp>

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

// MOGAC choice among labels with signed values `values` (the node's own label `current` wins ties).
MogacResult ChooseLabel(int current, const int *labels, const float *values, int count)
{
    MultiNarrowBand2D::Candidates candidates;
    MultiNarrowBand2D::Speeds updated;
    candidates.fill(MultiNarrowBand2D::kNoLabel);
    updated.fill(0.0f);
    for (int k = 0; k < count && k < MultiNarrowBand2D::kCandidates; k++) {
        candidates[k] = labels[k];
        updated[k] = values[k];
    }
    return MogacUpdate(current, candidates, updated);
}

// Labeled counterpart of Bilinear: each label at the cell's corners gets its bilinear signed value;
// the two-smallest rule picks the label and distance. The nearest corner's label wins ties.
MogacResult LabeledBilinear(const Block &block, float u, float v)
{
    if (block.isTile()) return {block.tileLabel, block.tileValue};
    const int a = std::clamp(static_cast<int>(std::floor(u)), 0, AdaptiveGrid2D::kBlockCells - 1);
    const int b = std::clamp(static_cast<int>(std::floor(v)), 0, AdaptiveGrid2D::kBlockCells - 1);
    const float fx = std::clamp(u - a, 0.0f, 1.0f);
    const float fy = std::clamp(v - b, 0.0f, 1.0f);
    const int nodes[4] = {Block::index(a, b), Block::index(a + 1, b), Block::index(a, b + 1), Block::index(a + 1, b + 1)};
    const float weights[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
    int labels[4];
    float values[4];
    int count = 0;
    for (int k = 0; k < 4; k++) {
        const int l = block.label(nodes[k]);
        if (std::find(labels, labels + count, l) != labels + count) continue;
        float s = 0.0f;
        for (int m = 0; m < 4; m++) s += weights[m] * block.signedValue(nodes[m], l);
        labels[count] = l;
        values[count++] = s;
    }
    const int nearest = nodes[(fx > 0.5f ? 1 : 0) + (fy > 0.5f ? 2 : 0)];
    return ChooseLabel(block.label(nearest), labels, values, count);
}
}  // namespace

void AdaptiveGrid2D::initializeLabeled(const ImageI &labels, const ImageF &distance, int maxDepth, float bandWidth)
{
    clear();
    mLabeled = true;
    mMaxDepth = std::clamp(maxDepth, 0, kMaxSupportedDepth);
    mBandWidth = bandWidth;
    const int W = labels.width();
    const int H = labels.height();
    mBlocksX = std::max(1, (W - 1 + kBlockCells - 1) / kBlockCells);
    mBlocksY = std::max(1, (H - 1 + kBlockCells - 1) / kBlockCells);
    mDomainMax = Vec2(static_cast<float>(mBlocksX * kBlockCells), static_cast<float>(mBlocksY * kBlockCells));
    for (int bj = 0; bj < mBlocksY; bj++) {
        for (int bi = 0; bi < mBlocksX; bi++) {
            Block block;
            block.labeled = true;
            block.key = Key{0, bi, bj};
            block.h = 1.0f;
            block.origin = Vec2(static_cast<float>(bi * kBlockCells), static_cast<float>(bj * kBlockCells));
            block.materialize();
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int x = std::min(bi * kBlockCells + a, W - 1);
                    const int y = std::min(bj * kBlockCells + b, H - 1);
                    block.phi[Block::index(a, b)] = std::min(std::abs(distance.at(x, y)), mBandWidth);
                    block.labels[Block::index(a, b)] = labels.at(x, y);
                }
            }
            allocate(block);
        }
    }
    rebuildLeafList();
    updateActive();
    compactTiles();
}

MogacResult AdaptiveGrid2D::sampleLabel(const Vec2 &q) const
{
    const Vec2 p(std::clamp(q.x(), 0.0f, mDomainMax.x()), std::clamp(q.y(), 0.0f, mDomainMax.y()));
    const Block &block = mBlocks[findLeaf(p)];
    const Vec2 u = (p - block.origin) / block.h;
    return LabeledBilinear(block, u.x(), u.y());
}

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
    mLabeled = false;
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
        child.labeled = parent.labeled;
        if (parent.isTile()) {
            child.makeTile(parent.tileValue, parent.tileLabel);
        } else if (parent.labeled) {
            child.materialize();
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const Vec2 u = (child.nodePosition(a, b) - parent.origin) / parent.h;
                    const MogacResult r = LabeledBilinear(parent, u.x(), u.y());
                    child.phi[Block::index(a, b)] = r.distance;
                    child.labels[Block::index(a, b)] = r.label;
                }
            }
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
    // Labeled grids only merge single-region blocks: a region boundary stays at the finer depth.
    if (mLabeled) {
        const int first = mBlocks[block.children[0]].label(0);
        for (int c : block.children) {
            const Block &child = mBlocks[c];
            if (child.isTile()) {
                if (child.tileLabel != first) return false;
                continue;
            }
            for (int l : child.labels) {
                if (l != first) return false;
            }
        }
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
    for (int c : children) {
        allTiles &= mBlocks[c].isTile() && mBlocks[c].tileValue == mBlocks[children[0]].tileValue && mBlocks[c].tileLabel == mBlocks[children[0]].tileLabel;
    }
    if (allTiles) {
        mBlocks[index].makeTile(mBlocks[children[0]].tileValue, mBlocks[children[0]].tileLabel);
    } else if (mLabeled) {
        std::vector<float> values(kNodeCount);
        std::vector<int> labels(kNodeCount);
        for (int b = 0; b < kBlockNodes; b++) {
            for (int a = 0; a < kBlockNodes; a++) {
                const MogacResult r = sampleLabel(mBlocks[index].nodePosition(a, b));
                values[Block::index(a, b)] = r.distance;
                labels[Block::index(a, b)] = r.label;
            }
        }
        mBlocks[index].phi = std::move(values);
        mBlocks[index].labels = std::move(labels);
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
    if (mLabeled) {
        synchronizeLabeled();
        return;
    }
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

void AdaptiveGrid2D::synchronizeLabeled()
{
    // Shared corners: the finest leaves' copies are combined per label (average signed value of
    // each label present) and the two-smallest rule picks the label and distance.
    for (size_t g = 0; g + 1 < mSyncGroupStart.size(); g++) {
        const int begin = mSyncGroupStart[g];
        const int end = mSyncGroupStart[g + 1];
        int depth = -1;
        for (int e = begin; e < end; e++) depth = std::max(depth, static_cast<int>(mSyncEntries[e].depth));
        int labels[4];
        float values[4];
        int count = 0, finest = 0, current = 0;
        for (int e = begin; e < end; e++) {
            if (mSyncEntries[e].depth != depth) continue;
            const Block &block = mBlocks[mSyncEntries[e].block];
            const int l = block.label(mSyncEntries[e].node);
            if (finest++ == 0) current = l;
            if (count < 4 && std::find(labels, labels + count, l) == labels + count) labels[count++] = l;
        }
        for (int k = 0; k < count; k++) {
            float sum = 0.0f;
            for (int e = begin; e < end; e++) {
                if (mSyncEntries[e].depth == depth) sum += mBlocks[mSyncEntries[e].block].signedValue(mSyncEntries[e].node, labels[k]);
            }
            values[k] = sum / finest;
        }
        const MogacResult r = ChooseLabel(current, labels, values, count);
        for (int e = begin; e < end; e++) {
            Block &block = mBlocks[mSyncEntries[e].block];
            if (block.isTile() && block.tileLabel == r.label && std::abs(block.tileValue - r.distance) <= kTileTolerance) continue;
            block.materialize();
            block.phi[mSyncEntries[e].node] = r.distance;
            block.labels[mSyncEntries[e].node] = r.label;
        }
    }
    // Coarse/fine borders: the fine leaf's in-between corners take the midpoint of their neighbors
    // along the border, per label as above (plain averaging when both share a label).
    for (const MidpointRule &rule : mMidpointRules) {
        Block &block = mBlocks[rule.block];
        if (block.isTile()) continue;
        const int la = block.labels[rule.nodeA], lb = block.labels[rule.nodeB];
        if (la == lb) {
            block.phi[rule.node] = 0.5f * (block.phi[rule.nodeA] + block.phi[rule.nodeB]);
            block.labels[rule.node] = la;
            continue;
        }
        const int labels[2] = {la, lb};
        const float values[2] = {0.5f * (block.signedValue(rule.nodeA, la) + block.signedValue(rule.nodeB, la)),
                                 0.5f * (block.signedValue(rule.nodeA, lb) + block.signedValue(rule.nodeB, lb))};
        const MogacResult r = ChooseLabel(std::min(la, lb), labels, values, 2);
        block.phi[rule.node] = r.distance;
        block.labels[rule.node] = r.label;
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
        if (uniform && block.labeled) {
            for (int l : block.labels) uniform &= l == block.labels[0];
            if (uniform) block.makeTile(first, block.labels[0]);
            continue;
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
                    // Same upwind scheme as LevelSet2D::speed, in this block's cell units.
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

void AdaptiveGrid2D::gatherLocalLabeled(int index, std::array<float, kGhostWidth * kGhostWidth> &distance,
                                        std::array<int, kGhostWidth * kGhostWidth> &labels) const
{
    const Block &block = mBlocks[index];
    const auto slotFor = [&](int a, int b) {
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
            const int l = (b + 1) * kGhostWidth + (a + 1);
            if (a >= 0 && b >= 0 && a < kBlockNodes && b < kBlockNodes) {
                distance[l] = block.value(Block::index(a, b));
                labels[l] = block.label(Block::index(a, b));
                continue;
            }
            const int neighbor = block.neighbors[slotFor(a, b)];
            if (neighbor < 0) {
                const int node = Block::index(std::clamp(a, 0, kBlockCells), std::clamp(b, 0, kBlockCells));
                distance[l] = block.value(node);
                labels[l] = block.label(node);
                continue;
            }
            const Block &src = mBlocks[neighbor];
            const Vec2 u = (block.nodePosition(a, b) - src.origin) / src.h;
            const MogacResult r = LabeledBilinear(src, u.x(), u.y());
            distance[l] = r.distance;
            labels[l] = r.label;
        }
    }
}

void AdaptiveGrid2D::evolveLabeled(int iterations, float curvatureWeight, float backgroundCurvatureWeight, const std::vector<NodeVelocities> &velocities,
                                   bool advectBackground)
{
    if (!mPlansValid) buildPlans();
    const std::vector<int> &active = mActiveLeaves;
    assert(velocities.size() == active.size());
    constexpr int kGhostCount = kGhostWidth * kGhostWidth;
    constexpr int kLayers = 3;
    constexpr int kStride[2] = {1, kGhostWidth};
    std::vector<std::array<float, kNodeCount>> nextDistance(active.size());
    std::vector<std::array<int, kNodeCount>> nextLabels(active.size());
    for (int it = 0; it < iterations; it++) {
        SPRINGLS_OMP(omp parallel for if(active.size() >= kParallelMinBlocks))
        for (int n = 0; n < static_cast<int>(active.size()); n++) {
            const int index = active[n];
            std::array<float, kGhostCount> dist;
            std::array<int, kGhostCount> lab;
            gatherLocalLabeled(index, dist, lab);
            const Block &block = mBlocks[index];
            const float h = block.h, invH = 1.0f / h;
            const auto L = [](int a, int b) { return (b + 1) * kGhostWidth + (a + 1); };
            // Signed value of `label` at local slot l, in cell units.
            const auto s = [&](int l, int label) { return ((lab[l] == label) ? -dist[l] : dist[l]) * invH; };
            std::array<float, kGhostCount> newDist = dist;
            std::array<int, kGhostCount> newLab = lab;
            // 1. Zero layer: the two-smallest rule over the candidate labels' updated values.
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int l = L(a, b);
                    if (dist[l] > 0.5f * h) continue;
                    MultiNarrowBand2D::Candidates candidates;
                    MultiNarrowBand2D::Speeds updated;
                    candidates.fill(MultiNarrowBand2D::kNoLabel);
                    updated.fill(0.0f);
                    int count = 0;
                    const int around[5] = {l, l - 1, l + 1, l - kGhostWidth, l + kGhostWidth};
                    const Vec2 g = velocities[n][Block::index(a, b)];
                    for (int k = 0; k < 5; k++) {
                        const int label = lab[around[k]];
                        if (std::find(candidates.begin(), candidates.begin() + count, label) != candidates.begin() + count) continue;
                        const float v11 = s(l, label);
                        const float v01 = s(l - 1, label), v21 = s(l + 1, label), v10 = s(l - kGhostWidth, label), v12 = s(l + kGhostWidth, label);
                        const float weight = (label == 0) ? backgroundCurvatureWeight : curvatureWeight;
                        float kappa = 0.0f;
                        if (weight != 0.0f) {
                            const float v00 = s(l - 1 - kGhostWidth, label), v20 = s(l + 1 - kGhostWidth, label);
                            const float v22 = s(l + 1 + kGhostWidth, label), v02 = s(l - 1 + kGhostWidth, label);
                            const float DxCtr = 0.5f * (v21 - v01), DyCtr = 0.5f * (v12 - v10);
                            const float DxxCtr = v21 - v11 - v11 + v01, DyyCtr = v12 - v11 - v11 + v10;
                            const float DxyCtr = (v22 - v02 - v20 + v00) * 0.25f;
                            const float numer = 0.5f * (DyCtr * DyCtr * DxxCtr - 2 * DxCtr * DyCtr * DxyCtr + DxCtr * DxCtr * DyyCtr);
                            const float denom = DxCtr * DxCtr + DyCtr * DyCtr;
                            kappa = (std::abs(denom) > 1E-5f) ? weight * numer / denom : 0.0f;
                            kappa = std::clamp(kappa, -kMaxCurvatureForce, kMaxCurvatureForce);
                        }
                        // Springl pulls move objects only (the background just smooths); a flow moves all.
                        float advection = 0.0f;
                        if (label != 0 || advectBackground) {
                            if (g.x() > 0) advection = g.x() * (v11 - v01);
                            else if (g.x() < 0) advection = g.x() * (v21 - v11);
                            if (g.y() > 0) advection += g.y() * (v11 - v10);
                            else if (g.y() < 0) advection += g.y() * (v12 - v11);
                        }
                        candidates[count] = label;
                        updated[count++] = v11 + kEvolveTimeStep * std::clamp(kappa - advection, -1.0f, 1.0f);
                    }
                    const MogacResult r = MogacUpdate(lab[l], candidates, updated);
                    newLab[l] = r.label;
                    newDist[l] = r.distance * h;
                }
            }
            // Ghosts outside the domain replicate a border corner; they are not neighbors.
            const auto inDomain = [&](int a, int b) {
                const Vec2 q = block.nodePosition(a, b);
                return q.x() >= 0.0f && q.y() >= 0.0f && q.x() <= mDomainMax.x() && q.y() <= mDomainMax.y();
            };
            // 2. Plug: a corner none of whose 8 neighbors shares its label joins the most common one.
            const std::array<int, kGhostCount> labelsBefore = newLab;
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int l = L(a, b);
                    std::array<int, 8> around;
                    int m = 0;
                    bool alone = true;
                    for (int db = -1; db <= 1; db++) {
                        for (int da = -1; da <= 1; da++) {
                            if ((da == 0 && db == 0) || !inDomain(a + da, b + db)) continue;
                            const int q = labelsBefore[l + da + db * kGhostWidth];
                            alone &= q != labelsBefore[l];
                            around[m++] = q;
                        }
                    }
                    if (!alone || m == 0) continue;
                    std::sort(around.begin(), around.begin() + m);
                    int bestLabel = around[0], bestCount = 0;
                    for (int x = 0; x < m;) {
                        int y = x;
                        while (y < m && around[y] == around[x]) y++;
                        if (y - x > bestCount) {
                            bestCount = y - x;
                            bestLabel = around[x];
                        }
                        x = y;
                    }
                    newLab[l] = bestLabel;
                }
            }
            // 3. The zero layer is what lies on an interface (a 4-neighbor of another label); any other
            //    corner a plug or a moving front left behind is rebuilt from its neighbors.
            std::array<uint8_t, kNodeCount> onInterface{};
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    const int l = L(a, b);
                    if (newDist[l] > 0.5f * h) continue;
                    bool interface = false;
                    for (int ax = 0; ax < 2; ax++) {
                        for (int sgn = -1; sgn <= 1; sgn += 2) {
                            const int da = (ax == 0) ? sgn : 0, db = (ax == 1) ? sgn : 0;
                            interface |= inDomain(a + da, b + db) && newLab[l + sgn * kStride[ax]] != newLab[l];
                        }
                    }
                    if (interface) {
                        onInterface[Block::index(a, b)] = 1;
                    } else {
                        newDist[l] = mBandWidth;
                    }
                }
            }
            // 4. Distance layers around the zero layer, one layer at a time from the previous one.
            for (int band = 1; band <= kLayers; band++) {
                const std::array<float, kGhostCount> prev = newDist;
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int a = 0; a < kBlockNodes; a++) {
                        if (onInterface[Block::index(a, b)]) continue;
                        const int l = L(a, b);
                        if (prev[l] <= (band - 0.5f) * h) continue;
                        const int label = newLab[l];
                        float best = mBandWidth;
                        for (int ax = 0; ax < 2; ax++) {
                            for (int sgn = -1; sgn <= 1; sgn += 2) {
                                const int q = l + sgn * kStride[ax];
                                const float sq = (newLab[q] == label) ? -prev[q] : prev[q];
                                best = std::min(best, std::abs(sq - h));
                            }
                        }
                        newDist[l] = best;
                    }
                }
            }
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    nextDistance[n][Block::index(a, b)] = std::clamp(newDist[L(a, b)], 0.0f, mBandWidth);
                    nextLabels[n][Block::index(a, b)] = newLab[L(a, b)];
                }
            }
        }
        for (size_t n = 0; n < active.size(); n++) {
            Block &block = mBlocks[active[n]];
            block.phi.assign(nextDistance[n].begin(), nextDistance[n].end());
            block.labels.assign(nextLabels[n].begin(), nextLabels[n].end());
        }
        synchronize();
    }
}

int AdaptiveGrid2D::correctLabels(const std::function<bool(const Vec2 &, float, int &, float &)> &measure)
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
                int label;
                float d;
                if (!measure(block.nodePosition(a, b), block.h, label, d)) continue;
                const int node = Block::index(a, b);
                if (block.labels[node] == label) continue;
                block.labels[node] = label;
                block.phi[node] = std::clamp(std::abs(d), 0.0f, mBandWidth);
                corrected++;
            }
        }
    }
    if (corrected > 0) synchronize();
    return corrected;
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
    extractContourOf([](const Block &block, int node) { return block.value(node); }, true, vertices, curves, vertexScales);
}

void AdaptiveGrid2D::extractLabeledContour(Points2 &vertices, Curves &curves, std::vector<int> &curveLabels, std::vector<float> &vertexScales,
                                           std::vector<uint8_t> *junctionVertices) const
{
    if (junctionVertices) junctionVertices->clear();
    vertices.clear();
    curves.clear();
    curveLabels.clear();
    vertexScales.clear();
    std::vector<int> objectLabels;
    for (int index : mLeaves) {
        const Block &block = mBlocks[index];
        if (block.isTile()) continue;  // one label: no boundary inside
        for (int l : block.labels) {
            if (l > 0) objectLabels.push_back(l);
        }
    }
    std::sort(objectLabels.begin(), objectLabels.end());
    objectLabels.erase(std::unique(objectLabels.begin(), objectLabels.end()), objectLabels.end());
    // Junction cells, on the block's own node lattice (boundary lines extend within the block).
    const auto junction = [](const Block &block, int a, int b, Vec2 &J) {
        Vec2 local;
        const bool found =
            FindJunction([&](int i, int j) { return block.label(Block::index(i, j)); },
                         [&](int i, int j, int l) { return block.signedValue(Block::index(i, j), l); }, kBlockCells + 1, kBlockCells + 1, a, b, local);
        if (found) J = block.origin + block.h * local;
        return found;
    };
    Points2 labelVertices;
    Curves labelCurves;
    std::vector<float> labelScales;
    std::vector<uint32_t> labelJunctions;
    for (int label : objectLabels) {
        extractContourOf([label](const Block &block, int node) { return block.signedValue(node, label); }, true, labelVertices, labelCurves, labelScales,
                         junction, &labelJunctions);
        const uint32_t offset = static_cast<uint32_t>(vertices.size());
        vertices.insert(vertices.end(), labelVertices.begin(), labelVertices.end());
        vertexScales.insert(vertexScales.end(), labelScales.begin(), labelScales.end());
        if (junctionVertices) {
            junctionVertices->resize(vertices.size(), 0);
            for (uint32_t v : labelJunctions) (*junctionVertices)[offset + v] = 1;
        }
        for (Curve &curve : labelCurves) {
            for (uint32_t &v : curve) v += offset;
            curves.push_back(std::move(curve));
            curveLabels.push_back(label);
        }
    }
}

void AdaptiveGrid2D::extractContourOf(const std::function<float(const Block &, int)> &value, bool skipTiles, Points2 &vertices, Curves &curves,
                                      std::vector<float> &vertexScales, const std::function<bool(const Block &, int, int, Vec2 &)> &junction,
                                      std::vector<uint32_t> *junctionVertices) const
{
    if (junctionVertices) junctionVertices->clear();
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
        if (skipTiles && block.isTile()) continue;  // a tile has one sign everywhere
        for (int b = 0; b < kBlockCells; b++) {
            for (int a = 0; a < kBlockCells; a++) {
                // Corners counter-clockwise: c0 (a,b), c1 (a+1,b), c2 (a+1,b+1), c3 (a,b+1).
                const int ca[4] = {a, a + 1, a + 1, a};
                const int cb[4] = {b, b, b + 1, b + 1};
                float val[4];
                Vec2 pos[4];
                for (int k = 0; k < 4; k++) {
                    val[k] = value(block, Block::index(ca[k], cb[k]));
                    // Push values away from zero (zero itself counts as outside).
                    if (std::abs(val[k]) < zeroEpsilon) val[k] = (val[k] < 0.0f) ? -zeroEpsilon : zeroEpsilon;
                    pos[k] = block.nodePosition(ca[k], cb[k]);
                }
                Vec2 cross[4];
                std::pair<uint32_t, uint32_t> cell[2];
                int cellCount = 0;
                MarchSquare(val, pos, cross, [&](int e0, int e1) {
                    const uint32_t u = vertexOf(cross[e0], block.h);
                    const uint32_t v = vertexOf(cross[e1], block.h);
                    if (u != v && cellCount < 2) cell[cellCount++] = {u, v};
                });
                Vec2 J;
                if (cellCount == 1 && junction && junction(block, a, b, J)) {
                    const uint32_t k = vertexOf(J, block.h);
                    if (k != cell[0].first && k != cell[0].second) {
                        if (junctionVertices) junctionVertices->push_back(k);
                        segments.emplace_back(cell[0].first, k);
                        segments.emplace_back(k, cell[0].second);
                        continue;
                    }
                }
                for (int n = 0; n < cellCount; n++) segments.push_back(cell[n]);
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
