#include <springls/AdaptiveGrid3D.hpp>
#include <springls/Profile.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace springls
{
namespace
{
// Pseudo-time step of evolve() in cell units (as in AdaptiveGrid2D).
constexpr float kEvolveTimeStep = 0.5f;
constexpr float kMaxCurvatureForce = 10.0f;
// A tile is re-materialized when a shared corner differs from its value by more than this.
constexpr float kTileTolerance = 1E-6f;
// Probes this far (cell units) outside a node or edge find the leaves around it.
constexpr float kProbeOffset = 0.25f;
// Local coordinates within this of an integer are a lattice corner.
constexpr float kLatticeTolerance = 1E-3f;

using Block = AdaptiveGrid3D::Block;
constexpr int N = AdaptiveGrid3D::kBlockCells;
constexpr float kHalf = 0.5f * N;

// Neighbor slot of a local position u (cell units) outside the block, or -1 inside it.
//   faces:   f * 4 + q1 + 2 q2 (f = 2 axis + side; q = quadrant along the cyclic other axes)
//   edges:   kFaceSlots + e * 2 + half (e = 4 t + side1 + 2 side2 for in-range axis t)
//   corners: kFaceSlots + kEdgeSlots + sx + 2 sy + 4 sz
int SlotOf(const Vec3 &u)
{
    int out[3], outCount = 0;
    for (int a = 0; a < 3; a++) {
        out[a] = (u[a] < 0.0f) ? 0 : ((u[a] > static_cast<float>(N)) ? 1 : -1);
        outCount += (out[a] >= 0);
    }
    if (outCount == 0) return -1;
    if (outCount == 1) {
        const int a = (out[0] >= 0) ? 0 : ((out[1] >= 0) ? 1 : 2);
        const int q1 = (u[(a + 1) % 3] < kHalf) ? 0 : 1, q2 = (u[(a + 2) % 3] < kHalf) ? 0 : 1;
        return (2 * a + out[a]) * 4 + q1 + 2 * q2;
    }
    if (outCount == 2) {
        const int t = (out[0] < 0) ? 0 : ((out[1] < 0) ? 1 : 2);
        const int e = 4 * t + out[(t + 1) % 3] + 2 * out[(t + 2) % 3];
        return AdaptiveGrid3D::kFaceSlots + 2 * e + ((u[t] < kHalf) ? 0 : 1);
    }
    return AdaptiveGrid3D::kFaceSlots + AdaptiveGrid3D::kEdgeSlots + out[0] + 2 * out[1] + 4 * out[2];
}

// A representative local position (cell units) of neighbor slot `slot`.
Vec3 SlotProbe(int slot)
{
    const float lo = -kProbeOffset, hi = N + kProbeOffset;
    const auto quarter = [](int q) { return (q == 0) ? 0.25f * N : 0.75f * N; };
    Vec3 u;
    if (slot < AdaptiveGrid3D::kFaceSlots) {
        const int f = slot / 4, a = f / 2, side = f % 2, q1 = slot % 2, q2 = (slot / 2) % 2;
        u[a] = side ? hi : lo;
        u[(a + 1) % 3] = quarter(q1);
        u[(a + 2) % 3] = quarter(q2);
    } else if (slot < AdaptiveGrid3D::kFaceSlots + AdaptiveGrid3D::kEdgeSlots) {
        const int s = slot - AdaptiveGrid3D::kFaceSlots, e = s / 2, half = s % 2, t = e / 4;
        u[t] = quarter(half);
        u[(t + 1) % 3] = (e & 1) ? hi : lo;
        u[(t + 2) % 3] = (e & 2) ? hi : lo;
    } else {
        const int c = slot - AdaptiveGrid3D::kFaceSlots - AdaptiveGrid3D::kEdgeSlots;
        for (int a = 0; a < 3; a++) u[a] = ((c >> a) & 1) ? hi : lo;
    }
    return u;
}

bool Inside(float v) { return v < 0.0f; }
}  // namespace

float Block::interpolate(const Vec3 &u) const
{
    if (isTile()) return tileValue;
    int i[3];
    float f[3];
    for (int a = 0; a < 3; a++) {
        i[a] = std::clamp(static_cast<int>(std::floor(u[a])), 0, N - 1);
        f[a] = std::clamp(u[a] - static_cast<float>(i[a]), 0.0f, 1.0f);
    }
    const auto v = [&](int da, int db, int dc) { return phi[index(i[0] + da, i[1] + db, i[2] + dc)]; };
    const float c00 = v(0, 0, 0) * (1 - f[0]) + v(1, 0, 0) * f[0];
    const float c10 = v(0, 1, 0) * (1 - f[0]) + v(1, 1, 0) * f[0];
    const float c01 = v(0, 0, 1) * (1 - f[0]) + v(1, 0, 1) * f[0];
    const float c11 = v(0, 1, 1) * (1 - f[0]) + v(1, 1, 1) * f[0];
    return (c00 * (1 - f[1]) + c10 * f[1]) * (1 - f[2]) + (c01 * (1 - f[1]) + c11 * f[1]) * f[2];
}

void AdaptiveGrid3D::clear()
{
    mBlocks.clear();
    mIndex.clear();
    mFreeList.clear();
    mLeaves.clear();
    mActiveLeaves.clear();
    mSyncGroupStart.clear();
    mSyncEntries.clear();
    mHangingRules.clear();
    mPlansValid = false;
    mBlocksPerAxis[0] = mBlocksPerAxis[1] = mBlocksPerAxis[2] = 0;
}

void AdaptiveGrid3D::initialize(const VolumeF &levelSet, int maxDepth, float bandWidth)
{
    clear();
    mMaxDepth = std::clamp(maxDepth, 0, kMaxSupportedDepth);
    mBandWidth = bandWidth;
    const Vec3i dims = levelSet.dimensions();
    for (int a = 0; a < 3; a++) {
        mBlocksPerAxis[a] = std::max(1, (dims[a] - 1 + N - 1) / N);
        mDomainMax[a] = static_cast<float>(mBlocksPerAxis[a] * N);
    }
    for (int bk = 0; bk < mBlocksPerAxis[2]; bk++) {
        for (int bj = 0; bj < mBlocksPerAxis[1]; bj++) {
            for (int bi = 0; bi < mBlocksPerAxis[0]; bi++) {
                Block block;
                block.key = Key{0, bi, bj, bk};
                block.h = 1.0f;
                block.origin = Vec3(static_cast<float>(bi * N), static_cast<float>(bj * N), static_cast<float>(bk * N));
                block.materialize();
                for (int c = 0; c < kBlockNodes; c++) {
                    for (int b = 0; b < kBlockNodes; b++) {
                        for (int a = 0; a < kBlockNodes; a++) {
                            const float v = levelSet.at(bi * N + a, bj * N + b, bk * N + c);
                            block.phi[Block::index(a, b, c)] = std::clamp(v, -mBandWidth, mBandWidth);
                        }
                    }
                }
                allocate(block);
            }
        }
    }
    rebuildLeafList();
    updateActive();
    compactTiles();
}

int AdaptiveGrid3D::allocate(const Block &block)
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

void AdaptiveGrid3D::release(int index)
{
    mIndex.erase(mBlocks[index].key);
    mBlocks[index].alive = false;
    mBlocks[index].makeTile(0.0f);
    mFreeList.push_back(index);
}

void AdaptiveGrid3D::rebuildLeafList()
{
    invalidatePlans();
    mLeaves.clear();
    for (int n = 0; n < static_cast<int>(mBlocks.size()); n++) {
        if (mBlocks[n].alive && mBlocks[n].isLeaf()) mLeaves.push_back(n);
    }
    std::sort(mLeaves.begin(), mLeaves.end(), [this](int a, int b) { return mBlocks[a].key < mBlocks[b].key; });
}

void AdaptiveGrid3D::restoreDerived()
{
    mIndex.clear();
    for (size_t n = 0; n < mBlocks.size(); n++) {
        if (mBlocks[n].alive) mIndex[mBlocks[n].key] = static_cast<int>(n);
    }
    rebuildLeafList();
    invalidatePlans();
}

int AdaptiveGrid3D::finestDepth() const
{
    int finest = 0;
    for (int index : mLeaves) finest = std::max(finest, mBlocks[index].key.depth);
    return finest;
}

int AdaptiveGrid3D::findLeaf(const Vec3 &q) const
{
    const Vec3 p = q.cwiseMax(Vec3::Zero()).cwiseMin(mDomainMax);
    int b[3];
    for (int a = 0; a < 3; a++) b[a] = std::clamp(static_cast<int>(std::floor(p[a] / N)), 0, mBlocksPerAxis[a] - 1);
    int index = mIndex.at(Key{0, b[0], b[1], b[2]});
    while (!mBlocks[index].isLeaf()) {
        const Block &block = mBlocks[index];
        const float half = 0.5f * block.size();
        int child = 0;
        for (int a = 0; a < 3; a++) child |= (p[a] >= block.origin[a] + half) ? (1 << a) : 0;
        index = block.children[child];
    }
    return index;
}

float AdaptiveGrid3D::sample(const Vec3 &q) const
{
    const Vec3 p = q.cwiseMax(Vec3::Zero()).cwiseMin(mDomainMax);
    const Block &block = mBlocks[findLeaf(p)];
    return block.interpolate((p - block.origin) / block.h);
}

bool AdaptiveGrid3D::refine(int index)
{
    if (!mBlocks[index].isLeaf()) return true;
    const int depth = mBlocks[index].key.depth;
    if (depth >= mMaxDepth) return false;
    // 2:1 balance: every leaf touching this block must be at least as fine before it splits.
    {
        const Vec3 center = mBlocks[index].origin + Vec3::Constant(0.5f * mBlocks[index].size());
        const float reach = 0.5f * mBlocks[index].size() + 0.5f * mBlocks[index].h;
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const Vec3 p = center + reach * Vec3(static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz));
                    if (!inDomain(p)) continue;
                    int neighbor = findLeaf(p);
                    while (mBlocks[neighbor].key.depth < depth) {
                        refine(neighbor);
                        neighbor = findLeaf(p);
                    }
                }
            }
        }
    }
    const Block parent = mBlocks[index];  // copy: allocate() may reallocate mBlocks
    for (int c = 0; c < 8; c++) {
        const Vec3 offset(static_cast<float>(c & 1), static_cast<float>((c >> 1) & 1), static_cast<float>((c >> 2) & 1));
        Block child;
        child.key = Key{depth + 1, 2 * parent.key.i + (c & 1), 2 * parent.key.j + ((c >> 1) & 1), 2 * parent.key.k + ((c >> 2) & 1)};
        child.h = 0.5f * parent.h;
        child.origin = parent.origin + 0.5f * parent.size() * offset;
        child.parent = index;
        child.active = parent.active;
        if (parent.isTile()) {
            child.makeTile(parent.tileValue);
        } else {
            child.materialize();
            for (int cc = 0; cc < kBlockNodes; cc++) {
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int a = 0; a < kBlockNodes; a++) {
                        child.phi[Block::index(a, b, cc)] = parent.interpolate((child.nodePosition(a, b, cc) - parent.origin) / parent.h);
                    }
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

bool AdaptiveGrid3D::canCoarsen(int index) const
{
    const Block &block = mBlocks[index];
    if (!block.alive || block.isLeaf()) return false;
    for (int c : block.children) {
        if (!mBlocks[c].isLeaf()) return false;
    }
    // Every leaf touching the block must be at most one depth finer than the merged block: probe
    // a shell of points just outside it, one per cell of the merged block.
    const float size = block.size();
    const float step = size / N;
    const float eps = 0.125f * step;
    float coords[N + 2];
    coords[0] = -eps;
    for (int s = 0; s < N; s++) coords[s + 1] = (s + 0.5f) * step;
    coords[N + 1] = size + eps;
    for (int z = 0; z < N + 2; z++) {
        for (int y = 0; y < N + 2; y++) {
            for (int x = 0; x < N + 2; x++) {
                const bool shell = x == 0 || y == 0 || z == 0 || x == N + 1 || y == N + 1 || z == N + 1;
                if (!shell) continue;
                const Vec3 p = block.origin + Vec3(coords[x], coords[y], coords[z]);
                if (!inDomain(p)) continue;
                if (mBlocks[findLeaf(p)].key.depth > block.key.depth + 1) return false;
            }
        }
    }
    return true;
}

void AdaptiveGrid3D::coarsen(int index)
{
    if (!canCoarsen(index)) return;
    // Restriction by injection: every coarse corner is a corner of one of the children.
    const std::array<int, 8> children = mBlocks[index].children;
    bool allTiles = true;
    for (int c : children) allTiles &= mBlocks[c].isTile() && mBlocks[c].tileValue == mBlocks[children[0]].tileValue;
    if (allTiles) {
        mBlocks[index].makeTile(mBlocks[children[0]].tileValue);
    } else {
        std::vector<float> values(kNodeCount);
        for (int c = 0; c < kBlockNodes; c++) {
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) values[Block::index(a, b, c)] = sample(mBlocks[index].nodePosition(a, b, c));
            }
        }
        mBlocks[index].phi = std::move(values);
    }
    bool anyActive = false;
    for (int c : children) {
        anyActive |= mBlocks[c].active;
        release(c);
    }
    mBlocks[index].children.fill(-1);
    mBlocks[index].active = anyActive;
    rebuildLeafList();
}

int AdaptiveGrid3D::leafAt(int index, const Vec3 &u) const
{
    const int slot = SlotOf(u);
    return (slot < 0) ? index : mBlocks[index].neighbors[slot];
}

void AdaptiveGrid3D::buildPlans()
{
    ProfileScope profileScope("grid: buildPlans");
    const int L = static_cast<int>(mLeaves.size());
    // Neighbor slots of every leaf.
    SPRINGLS_OMP(omp parallel for if (L >= kParallelMinBlocks))
    for (int n = 0; n < L; n++) {
        Block &block = mBlocks[mLeaves[n]];
        for (int s = 0; s < kNeighborSlots; s++) {
            const Vec3 p = block.origin + block.h * SlotProbe(s);
            block.neighbors[s] = inDomain(p) ? findLeaf(p) : -1;
        }
    }
    // Shared-corner groups (built once, by the lowest-indexed leaf owning the corner) and hanging
    // corners, per leaf, then concatenated in leaf order.
    std::vector<std::vector<SyncEntry>> groups(L);
    std::vector<std::vector<int>> groupSizes(L);
    std::vector<std::vector<HangingRule>> hanging(L);
    SPRINGLS_OMP(omp parallel for if (L >= kParallelMinBlocks))
    for (int n = 0; n < L; n++) {
        const int index = mLeaves[n];
        const Block &block = mBlocks[index];
        for (int c = 0; c < kBlockNodes; c++) {
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    if (a != 0 && b != 0 && c != 0 && a != N && b != N && c != N) continue;
                    const Vec3 u(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c));
                    const Vec3 p = block.nodePosition(a, b, c);
                    int leaves[8], count = 0, coarsest = block.key.depth;
                    for (int k = 0; k < 8; k++) {
                        const Vec3 d((k & 1) ? kProbeOffset : -kProbeOffset, (k & 2) ? kProbeOffset : -kProbeOffset, (k & 4) ? kProbeOffset : -kProbeOffset);
                        const int leaf = leafAt(index, u + d);
                        if (leaf < 0 || std::find(leaves, leaves + count, leaf) != leaves + count) continue;
                        leaves[count++] = leaf;
                        coarsest = std::min(coarsest, mBlocks[leaf].key.depth);
                    }
                    // Corner groups.
                    SyncEntry entries[8];
                    int entryCount = 0, owner = std::numeric_limits<int>::max();
                    for (int k = 0; k < count; k++) {
                        const Block &other = mBlocks[leaves[k]];
                        const Vec3 v = (p - other.origin) / other.h;
                        int ia[3];
                        bool corner = true;
                        for (int ax = 0; ax < 3; ax++) {
                            ia[ax] = static_cast<int>(std::lround(v[ax]));
                            corner &= std::abs(v[ax] - ia[ax]) <= kLatticeTolerance && ia[ax] >= 0 && ia[ax] <= N;
                        }
                        if (!corner) continue;
                        entries[entryCount++] = {leaves[k], static_cast<uint16_t>(Block::index(ia[0], ia[1], ia[2])), static_cast<uint8_t>(other.key.depth)};
                        owner = std::min(owner, leaves[k]);
                    }
                    if (entryCount > 1 && owner == index) {
                        groups[n].insert(groups[n].end(), entries, entries + entryCount);
                        groupSizes[n].push_back(entryCount);
                    }
                    // Hanging corners: odd corners of a fine leaf touching a coarser one.
                    if (coarsest < block.key.depth && ((a | b | c) & 1)) {
                        HangingRule rule{index, static_cast<uint16_t>(Block::index(a, b, c)), 0, {}};
                        const int odd[3] = {a & 1, b & 1, c & 1};
                        for (int k = 0; k < 4; k++) {
                            int s[3] = {a, b, c};
                            bool valid = true;
                            int bit = 0;
                            for (int ax = 0; ax < 3; ax++) {
                                if (!odd[ax]) continue;
                                s[ax] += ((k >> bit) & 1) ? 1 : -1;
                                bit++;
                            }
                            if (k >= (1 << bit)) valid = false;
                            if (valid) rule.sources[rule.count++] = static_cast<uint16_t>(Block::index(s[0], s[1], s[2]));
                        }
                        hanging[n].push_back(rule);
                    }
                }
            }
        }
    }
    mSyncGroupStart.clear();
    mSyncEntries.clear();
    mHangingRules.clear();
    for (int n = 0; n < L; n++) {
        size_t e = 0;
        for (int size : groupSizes[n]) {
            mSyncGroupStart.push_back(static_cast<int>(mSyncEntries.size()));
            mSyncEntries.insert(mSyncEntries.end(), groups[n].begin() + e, groups[n].begin() + e + size);
            e += size;
        }
        mHangingRules.insert(mHangingRules.end(), hanging[n].begin(), hanging[n].end());
    }
    mSyncGroupStart.push_back(static_cast<int>(mSyncEntries.size()));
    mPlansValid = true;
}

void AdaptiveGrid3D::synchronize()
{
    ProfileScope profileScope("grid: synchronize");
    if (!mPlansValid) buildPlans();
    // Shared corners take the average of the finest leaves that have them: values per group in
    // parallel, then the tiles that must change are materialized (allocation, sequential), then the
    // values are written in parallel (every corner belongs to one group).
    const int G = static_cast<int>(mSyncGroupStart.size()) - 1;
    std::vector<float> values(std::max(G, 0));
    SPRINGLS_OMP(omp parallel for if (G >= kParallelMinLightItems))
    for (int g = 0; g < G; g++) {
        const int begin = mSyncGroupStart[g], end = mSyncGroupStart[g + 1];
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
        values[g] = static_cast<float>(sum / count);
    }
    for (int g = 0; g < G; g++) {
        for (int e = mSyncGroupStart[g]; e < mSyncGroupStart[g + 1]; e++) {
            Block &block = mBlocks[mSyncEntries[e].block];
            if (block.isTile() && std::abs(block.tileValue - values[g]) > kTileTolerance) block.materialize();
        }
    }
    SPRINGLS_OMP(omp parallel for if (G >= kParallelMinLightItems))
    for (int g = 0; g < G; g++) {
        for (int e = mSyncGroupStart[g]; e < mSyncGroupStart[g + 1]; e++) {
            Block &block = mBlocks[mSyncEntries[e].block];
            if (!block.isTile()) block.phi[mSyncEntries[e].node] = values[g];
        }
    }
    // On a coarse/fine face or edge the fine leaf's odd corners interpolate the coarse ones (which
    // are never odd corners themselves, so the rules are independent).
    const int H = static_cast<int>(mHangingRules.size());
    SPRINGLS_OMP(omp parallel for if (H >= kParallelMinLightItems))
    for (int r = 0; r < H; r++) {
        const HangingRule &rule = mHangingRules[r];
        Block &block = mBlocks[rule.block];
        if (block.isTile()) continue;  // a tile is constant already
        float sum = 0.0f;
        for (int k = 0; k < rule.count; k++) sum += block.phi[rule.sources[k]];
        block.phi[rule.node] = sum / static_cast<float>(rule.count);
    }
}

void AdaptiveGrid3D::gatherLocal(int index, Local &local) const
{
    const Block &block = mBlocks[index];
    for (int c = -1; c <= kBlockNodes; c++) {
        for (int b = -1; b <= kBlockNodes; b++) {
            for (int a = -1; a <= kBlockNodes; a++) {
                float &out = local[((c + 1) * kGhostWidth + (b + 1)) * kGhostWidth + (a + 1)];
                if (a >= 0 && b >= 0 && c >= 0 && a < kBlockNodes && b < kBlockNodes && c < kBlockNodes) {
                    out = block.at(a, b, c);
                    continue;
                }
                const int neighbor = leafAt(index, Vec3(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)));
                if (neighbor < 0) {
                    // Outside the domain: replicate the border corner.
                    out = block.at(std::clamp(a, 0, N), std::clamp(b, 0, N), std::clamp(c, 0, N));
                    continue;
                }
                const Block &src = mBlocks[neighbor];
                if (src.key.depth == block.key.depth) {
                    // Same depth: the ghost corner is one of the neighbor's own corners.
                    const int sa = a - static_cast<int>(std::lround((src.origin.x() - block.origin.x()) / block.h));
                    const int sb = b - static_cast<int>(std::lround((src.origin.y() - block.origin.y()) / block.h));
                    const int sc = c - static_cast<int>(std::lround((src.origin.z() - block.origin.z()) / block.h));
                    out = src.at(sa, sb, sc);
                } else {
                    out = src.interpolate((block.nodePosition(a, b, c) - src.origin) / src.h);
                }
            }
        }
    }
}

void AdaptiveGrid3D::updateActive()
{
    ProfileScope profileScope("grid: updateActive");
    mActiveLeaves.clear();
    for (int index : mLeaves) {
        Block &block = mBlocks[index];
        const float threshold = activeThreshold(block.h);
        float minAbs = std::abs(block.tileValue);
        if (!block.isTile()) {
            minAbs = std::numeric_limits<float>::max();
            for (float v : block.phi) minAbs = std::min(minAbs, std::abs(v));
        }
        block.active = minAbs < threshold;
        if (block.active) mActiveLeaves.push_back(index);
    }
}

void AdaptiveGrid3D::compactTiles()
{
    for (int index : mLeaves) {
        Block &block = mBlocks[index];
        if (block.isTile() || block.active) continue;
        const float first = block.phi[0];
        if (std::abs(first) < mBandWidth) continue;
        if (std::all_of(block.phi.begin(), block.phi.end(), [first](float v) { return v == first; })) block.makeTile(first);
    }
}

void AdaptiveGrid3D::evolve(int iterations, float curvatureWeight, const VelocityFunction &velocity)
{
    if (!mPlansValid) buildPlans();
    const std::vector<int> &active = mActiveLeaves;
    const int A = static_cast<int>(active.size());
    // Pull velocities are computed only where the zero layer reaches, once per node.
    std::vector<NodeVelocities> velocities(active.size());
    std::vector<std::vector<uint8_t>> haveVelocity(active.size(), std::vector<uint8_t>(kNodeCount, 0));
    std::vector<std::array<float, kNodeCount>> next(active.size());
    constexpr int kLayers = 3;
    for (int it = 0; it < iterations; it++) {
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (A >= kParallelMinBlocks))
        for (int n = 0; n < A; n++) {
            const int index = active[n];
            const Block &block = mBlocks[index];
            const float h = block.h, invH = 1.0f / h;
            Local local;
            gatherLocal(index, local);
            const auto L = [](int a, int b, int c) { return ((c + 1) * kGhostWidth + (b + 1)) * kGhostWidth + (a + 1); };
            constexpr int kStride[3] = {1, kGhostWidth, kGhostWidth * kGhostWidth};
            // 1. The zero layer (|phi| <= h / 2) moves; everything else is rebuilt from it below
            //    (the sparse-field method, as NarrowBand3D does on the uniform grid).
            Local updated = local;
            std::array<uint8_t, kNodeCount> zeroLayer{};
            for (int c = 0; c < kBlockNodes; c++) {
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int a = 0; a < kBlockNodes; a++) {
                        const int l = L(a, b, c), node = Block::index(a, b, c);
                        if (std::abs(local[l]) > 0.5f * h) continue;
                        zeroLayer[node] = 1;
                        const auto v = [&](int da, int db, int dc) { return local[l + da * kStride[0] + db * kStride[1] + dc * kStride[2]] * invH; };
                        const float v111 = v(0, 0, 0);
                        const float neg[3] = {v111 - v(-1, 0, 0), v111 - v(0, -1, 0), v111 - v(0, 0, -1)};
                        const float pos[3] = {v(1, 0, 0) - v111, v(0, 1, 0) - v111, v(0, 0, 1) - v111};
                        float kappa = 0.0f;
                        if (curvatureWeight != 0.0f) {
                            const float v011 = v(-1, 0, 0), v211 = v(1, 0, 0), v101 = v(0, -1, 0), v121 = v(0, 1, 0), v110 = v(0, 0, -1), v112 = v(0, 0, 1);
                            const float DxCtr = 0.5f * (v211 - v011), DyCtr = 0.5f * (v121 - v101), DzCtr = 0.5f * (v112 - v110);
                            const float DxxCtr = v211 - 2 * v111 + v011, DyyCtr = v121 - 2 * v111 + v101, DzzCtr = v112 - 2 * v111 + v110;
                            const float DxyCtr = 0.25f * (v(1, 1, 0) - v(-1, 1, 0) - v(1, -1, 0) + v(-1, -1, 0));
                            const float DxzCtr = 0.25f * (v(1, 0, 1) - v(-1, 0, 1) - v(1, 0, -1) + v(-1, 0, -1));
                            const float DyzCtr = 0.25f * (v(0, 1, 1) - v(0, -1, 1) - v(0, 1, -1) + v(0, -1, -1));
                            const float numer =
                                0.5f * ((DyyCtr + DzzCtr) * DxCtr * DxCtr + (DxxCtr + DzzCtr) * DyCtr * DyCtr + (DxxCtr + DyyCtr) * DzCtr * DzCtr -
                                        2 * DxCtr * DyCtr * DxyCtr - 2 * DxCtr * DzCtr * DxzCtr - 2 * DyCtr * DzCtr * DyzCtr);
                            const float denom = DxCtr * DxCtr + DyCtr * DyCtr + DzCtr * DzCtr;
                            kappa = (std::abs(denom) > 1E-5f) ? curvatureWeight * numer / denom : 0.0f;
                            kappa = std::clamp(kappa, -kMaxCurvatureForce, kMaxCurvatureForce);
                        }
                        if (!haveVelocity[n][node]) {
                            velocities[n][node] = velocity(block.nodePosition(a, b, c), h);
                            haveVelocity[n][node] = 1;
                        }
                        const Vec3 &g = velocities[n][node];
                        float advection = 0.0f;
                        for (int ax = 0; ax < 3; ax++) {
                            if (g[ax] > 0) advection += g[ax] * neg[ax];
                            else if (g[ax] < 0) advection += g[ax] * pos[ax];
                        }
                        // At most half a cell per iteration, like the uniform narrow band.
                        updated[l] = (v111 + kEvolveTimeStep * std::clamp(kappa - advection, -1.0f, 1.0f)) * h;
                    }
                }
            }
            // 2. Distance layers 1..kLayers around the zero layer (in place; neighbor leaves' layers
            //    are a sweep behind across block borders until the next iteration).
            for (int band = 1; band <= kLayers; band++) {
                for (int c = 0; c < kBlockNodes; c++) {
                    for (int b = 0; b < kBlockNodes; b++) {
                        for (int a = 0; a < kBlockNodes; a++) {
                            if (zeroLayer[Block::index(a, b, c)]) continue;
                            const int l = L(a, b, c);
                            const float old = updated[l];
                            float v = old;
                            if (v < (-band + 0.5f) * h) {
                                v = -mBandWidth;
                                for (int ax = 0; ax < 3; ax++) {
                                    for (int s = -1; s <= 1; s += 2) {
                                        const float u = updated[l + s * kStride[ax]];
                                        if (u <= h) v = std::max(v, u);
                                    }
                                }
                                v -= h;
                            } else if (v > (band - 0.5f) * h) {
                                v = mBandWidth;
                                for (int ax = 0; ax < 3; ax++) {
                                    for (int s = -1; s <= 1; s += 2) {
                                        const float u = updated[l + s * kStride[ax]];
                                        if (u >= -h) v = std::min(v, u);
                                    }
                                }
                                v += h;
                            }
                            updated[l] = (old * v > 0.0f) ? v : old;
                        }
                    }
                }
            }
            // 3. Plug: a zero-layer node whose six neighbors all lie on its side is not on the interface.
            for (int c = 0; c < kBlockNodes; c++) {
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int a = 0; a < kBlockNodes; a++) {
                        const int l = L(a, b, c);
                        const float v = updated[l];
                        const float sgn = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
                        if (sgn == 0.0f || std::abs(v) >= 0.5f * h) continue;
                        bool plugged = true;
                        for (int ax = 0; ax < 3 && plugged; ax++) {
                            for (int s = -1; s <= 1; s += 2) plugged &= sgn * updated[l + s * kStride[ax]] > 0.0f;
                        }
                        if (plugged) updated[l] = sgn * h;
                    }
                }
            }
            for (int c = 0; c < kBlockNodes; c++) {
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int a = 0; a < kBlockNodes; a++) next[n][Block::index(a, b, c)] = std::clamp(updated[L(a, b, c)], -mBandWidth, mBandWidth);
                }
            }
        }
        for (int n = 0; n < A; n++) mBlocks[active[n]].phi.assign(next[n].begin(), next[n].end());
        synchronize();
    }
}

int AdaptiveGrid3D::correctSigns(const std::function<bool(const Vec3 &, float, float &)> &measure)
{
    const std::vector<int> &active = mActiveLeaves;
    const int A = static_cast<int>(active.size());
    int corrected = 0;
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) reduction(+ : corrected) if (A >= kParallelMinBlocks))
    for (int n = 0; n < A; n++) {
        Block &block = mBlocks[active[n]];
        if (block.isTile()) continue;
        for (int c = 0; c < kBlockNodes; c++) {
            for (int b = 0; b < kBlockNodes; b++) {
                for (int a = 0; a < kBlockNodes; a++) {
                    float &v = block.phi[Block::index(a, b, c)];
                    float d;
                    if (!measure(block.nodePosition(a, b, c), block.h, d)) continue;
                    if (Inside(d) == Inside(v)) continue;
                    v = std::clamp(d, -mBandWidth, mBandWidth);
                    corrected++;
                }
            }
        }
    }
    // Shared corners are measured identically by every leaf that has them (same position).
    if (corrected > 0) synchronize();
    return corrected;
}

void AdaptiveGrid3D::extractSurface(ElementType type, SurfaceMesh3D &mesh, std::vector<float> &vertexScales)
{
    ProfileScope profileScope("grid: extractSurface");
    if (!mPlansValid) buildPlans();  // neighbor slots find the cells across block borders
    mesh.clear();
    mesh.type = type;
    vertexScales.clear();
    // 1. Per leaf, in parallel: the crossed minimal edges it owns, each with the four cells around it.
    struct Edge {
        int leaves[4];
        Vec3i cells[4];
        bool inside0;  // phi < 0 at the edge's first corner
    };
    const int L = static_cast<int>(mLeaves.size());
    std::vector<std::vector<Edge>> leafEdges(L);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (L >= kParallelMinBlocks))
    for (int n = 0; n < L; n++) {
        const int leaf = mLeaves[n];
        const Block &block = mBlocks[leaf];
        if (block.isTile()) continue;  // a tile has one sign everywhere
        for (int a = 0; a < 3; a++) {
            const int ua = (a + 1) % 3, va = (a + 2) % 3;
            for (int c = 0; c < kBlockNodes; c++) {
                for (int b = 0; b < kBlockNodes; b++) {
                    for (int e = 0; e < kBlockNodes; e++) {
                        Vec3i p0(e, b, c);
                        if (p0[a] >= N) continue;
                        Vec3i p1 = p0;
                        p1[a]++;
                        const float f0 = block.at(p0.x(), p0.y(), p0.z()), f1 = block.at(p1.x(), p1.y(), p1.z());
                        if (Inside(f0) == Inside(f1)) continue;
                        // The four cells around the edge, in the uniform surface-nets order.
                        Vec3 mid = p0.cast<float>();
                        mid[a] += 0.5f;
                        Edge edge;
                        edge.inside0 = Inside(f0);
                        bool valid = true;
                        for (int k = 0; k < 4 && valid; k++) {
                            Vec3 u = mid;
                            u[ua] += (k & 2) ? -0.5f : 0.5f;
                            u[va] += (k & 1) ? -0.5f : 0.5f;
                            const int other = leafAt(leaf, u);
                            if (other < 0) {
                                valid = false;
                                break;
                            }
                            const Block &ob = mBlocks[other];
                            if (ob.key.depth > block.key.depth) valid = false;  // a finer cell subdivides this edge
                            const Vec3 w = (block.origin + block.h * u - ob.origin) / ob.h;
                            for (int ax = 0; ax < 3; ax++) edge.cells[k][ax] = std::clamp(static_cast<int>(std::floor(w[ax])), 0, N - 1);
                            edge.leaves[k] = other;
                        }
                        if (!valid) continue;
                        // Emitted once: by the lowest-indexed leaf among the same-depth ones sharing the edge.
                        int owner = leaf;
                        for (int k = 0; k < 4; k++) {
                            if (mBlocks[edge.leaves[k]].key.depth == block.key.depth) owner = std::min(owner, edge.leaves[k]);
                        }
                        if (owner == leaf) leafEdges[n].push_back(edge);
                    }
                }
            }
        }
    }
    // 2. In leaf order (as a sequential pass would): one vertex per crossed cell, numbered by first
    // use, and the polygons around each edge.
    std::unordered_map<uint64_t, uint32_t> cellVertex;
    std::vector<std::pair<int, Vec3i>> vertexCells;  // (leaf, cell) of each vertex
    const auto vertexOf = [&](int leaf, const Vec3i &cell) {
        const uint64_t key = (static_cast<uint64_t>(leaf) << 10) | static_cast<uint64_t>((cell.z() * N + cell.y()) * N + cell.x());
        const auto [it, inserted] = cellVertex.try_emplace(key, static_cast<uint32_t>(vertexCells.size()));
        if (inserted) vertexCells.emplace_back(leaf, cell);
        return it->second;
    };
    std::vector<std::array<uint32_t, 4>> polygons;  // fourth index == first for triangles
    for (int n = 0; n < L; n++) {
        for (const Edge &edge : leafEdges[n]) {
            uint32_t v[4];
            for (int k = 0; k < 4; k++) v[k] = vertexOf(edge.leaves[k], edge.cells[k]);
            // (v0, v2, v3, v1) circles the edge counterclockwise about +e_a; the outward
            // normal points from phi < 0 to phi > 0.
            const std::array<uint32_t, 4> cycle =
                edge.inside0 ? std::array<uint32_t, 4>{v[0], v[2], v[3], v[1]} : std::array<uint32_t, 4>{v[0], v[1], v[3], v[2]};
            std::array<uint32_t, 4> unique;
            int count = 0;
            for (int k = 0; k < 4; k++) {
                if (cycle[k] != cycle[(k + 3) % 4]) unique[count++] = cycle[k];
            }
            if (count == 4) {
                polygons.push_back(unique);
            } else if (count == 3) {
                polygons.push_back({unique[0], unique[1], unique[2], unique[0]});
            }
        }
    }
    // 3. Vertex positions, in parallel: the mean of the crossings on the cell's own twelve edges.
    const int V = static_cast<int>(vertexCells.size());
    mesh.vertexes.resize(V);
    vertexScales.resize(V);
    SPRINGLS_OMP(omp parallel for if (V >= kParallelMinHeavyItems))
    for (int id = 0; id < V; id++) {
        const auto &[leaf, cell] = vertexCells[id];
        const Block &block = mBlocks[leaf];
        Vec3 sum = Vec3::Zero();
        int count = 0;
        for (int a = 0; a < 3; a++) {
            for (int k = 0; k < 4; k++) {
                Vec3i p0 = cell;
                p0[(a + 1) % 3] += k & 1;
                p0[(a + 2) % 3] += (k >> 1) & 1;
                Vec3i p1 = p0;
                p1[a]++;
                const float f0 = block.at(p0.x(), p0.y(), p0.z()), f1 = block.at(p1.x(), p1.y(), p1.z());
                if (Inside(f0) == Inside(f1)) continue;
                const float t = (std::abs(f1 - f0) > 1E-12f) ? std::clamp(f0 / (f0 - f1), 0.0f, 1.0f) : 0.5f;
                Vec3 q = p0.cast<float>();
                q[a] += t;
                sum += q;
                count++;
            }
        }
        const Vec3 local = (count > 0) ? Vec3(sum / static_cast<float>(count)) : Vec3(cell.cast<float>() + Vec3::Constant(0.5f));
        mesh.vertexes[id] = block.origin + block.h * local;
        vertexScales[id] = block.h;
    }
    const Points3 &p = mesh.vertexes;
    for (const auto &poly : polygons) {
        const bool triangle = poly[3] == poly[0];
        if (type == ElementType::Triangle) {
            if (triangle) {
                mesh.indexes.insert(mesh.indexes.end(), {poly[0], poly[1], poly[2]});
            } else if (distanceSqr(p[poly[0]], p[poly[2]]) <= distanceSqr(p[poly[1]], p[poly[3]])) {
                mesh.indexes.insert(mesh.indexes.end(), {poly[0], poly[1], poly[2], poly[0], poly[2], poly[3]});
            } else {
                mesh.indexes.insert(mesh.indexes.end(), {poly[0], poly[1], poly[3], poly[1], poly[2], poly[3]});
            }
        } else if (!triangle) {
            mesh.indexes.insert(mesh.indexes.end(), poly.begin(), poly.end());
        } else {
            // A quad from the triangle: the midpoint of its longest edge becomes the fourth corner.
            int longest = 0;
            float best = -1.0f;
            for (int k = 0; k < 3; k++) {
                const float d = distanceSqr(p[poly[k]], p[poly[(k + 1) % 3]]);
                if (d > best) {
                    best = d;
                    longest = k;
                }
            }
            const uint32_t a = poly[longest], b = poly[(longest + 1) % 3], c = poly[(longest + 2) % 3];
            const uint32_t m = static_cast<uint32_t>(mesh.vertexes.size());
            mesh.vertexes.push_back(0.5f * (mesh.vertexes[a] + mesh.vertexes[b]));
            vertexScales.push_back(std::min(vertexScales[a], vertexScales[b]));
            mesh.indexes.insert(mesh.indexes.end(), {a, m, b, c});
        }
    }
    mesh.updateVertexNormals();
}

size_t AdaptiveGrid3D::tileCount() const
{
    size_t tiles = 0;
    for (int index : mLeaves) tiles += mBlocks[index].isTile() ? 1 : 0;
    return tiles;
}
}  // namespace springls
