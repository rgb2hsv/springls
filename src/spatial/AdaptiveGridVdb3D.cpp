// AdaptiveGrid3D::saveVdb / loadVdb / writeVdb / readVdb: the octree as OpenVDB grids (see the header for the
// layout), through VdbWriter / VdbReader.
#include <io/Vdb.hpp>
#include <spatial/AdaptiveGrid3D.hpp>

#include <algorithm>
#include <bit>
#include <bitset>
#include <cmath>
#include <istream>
#include <map>
#include <ostream>

namespace springls
{
namespace
{
constexpr int N = AdaptiveGrid3D::kBlockCells;
constexpr int M = AdaptiveGrid3D::kBlockNodes;
constexpr int kLeafValues = N * N * N;  // a VDB leaf node (8^3 voxels) holds one block's own corners
constexpr int32_t kVdbVersion = 1;
const std::string kMetaPrefix = "springls.meta.";

// Per-block flags in "blocks_d<d>".
enum : int32_t {
    kPresent = 1,
    kActive = 2,      // Block::active
    kActiveLeaf = 4,  // listed in activeLeaves()
    kTile = 8,        // value in "tiles_d<d>"
    kParity = 16,     // every corner in "phi_d<d>_p<n>"
};

struct Coord {
    int x = 0, y = 0, z = 0;
    bool operator==(const Coord &o) const { return x == o.x && y == o.y && z == o.z; }
    bool operator<(const Coord &o) const
    {
        if (z != o.z) return z < o.z;
        if (y != o.y) return y < o.y;
        return x < o.x;
    }
};
struct CoordHash {
    size_t operator()(const Coord &c) const
    {
        return (static_cast<size_t>(static_cast<uint32_t>(c.x)) * 73856093u) ^ (static_cast<size_t>(static_cast<uint32_t>(c.y)) * 19349663u) ^
               (static_cast<size_t>(static_cast<uint32_t>(c.z)) * 83492791u);
    }
};

Coord LeafOf(const Coord &p) { return {p.x >> 3, p.y >> 3, p.z >> 3}; }
int SlotOf(const Coord &p) { return ((p.x & 7) << 6) | ((p.y & 7) << 3) | (p.z & 7); }
Coord Corner(const AdaptiveGrid3D::Key &key, int a, int b, int c) { return {N * key.i + a, N * key.j + b, N * key.k + c}; }
VdbCoord ToVdb(const Coord &c) { return {c.x, c.y, c.z}; }
int Parity(const AdaptiveGrid3D::Key &key) { return (key.i & 1) | ((key.j & 1) << 1) | ((key.k & 1) << 2); }
std::string DepthName(const std::string &prefix, const char *kind, int depth) { return prefix + kind + std::to_string(depth); }
std::string ParityName(const std::string &prefix, int depth, int parity) { return DepthName(prefix, "phi_d", depth) + "_p" + std::to_string(parity); }

// Active voxels of one grid being written.
template <class T>
struct Voxels {
    std::vector<VdbCoord> coords;
    std::vector<T> values;
    void add(const Coord &p, T v)
    {
        coords.push_back(ToVdb(p));
        values.push_back(v);
    }
};
}  // namespace

bool AdaptiveGrid3D::writeVdb(VdbWriter &writer, const std::string &prefix, std::string &error) const
{
    std::vector<int> order;
    for (int n = 0; n < static_cast<int>(mBlocks.size()); n++) {
        if (mBlocks[n].alive) order.push_back(n);
    }
    if (order.empty()) {
        error = "the grid is empty";
        return false;
    }
    std::sort(order.begin(), order.end(), [this](int a, int b) { return mBlocks[a].key < mBlocks[b].key; });
    std::vector<uint8_t> inActiveList(mBlocks.size(), 0);
    for (int index : mActiveLeaves) {
        if (index >= 0 && index < static_cast<int>(mBlocks.size())) inActiveList[index] = 1;
    }

    for (size_t begin = 0; begin < order.size();) {
        const int depth = mBlocks[order[begin]].key.depth;
        size_t end = begin;
        while (end < order.size() && mBlocks[order[end]].key.depth == depth) end++;
        const double h = std::ldexp(1.0, -depth);

        // Corners by leaf node: each materialized block's own (a, b, c < 8) corners first, then its far-face
        // corners where no block stored one. A far-face corner that differs from the stored one puts the
        // whole block in a parity grid.
        struct Leaf {
            std::array<float, kLeafValues> values{};
            std::bitset<kLeafValues> set;
        };
        std::unordered_map<Coord, Leaf, CoordHash> leaves;
        std::vector<uint8_t> parity(end - begin, 0);
        for (size_t n = begin; n < end; n++) {
            const Block &block = mBlocks[order[n]];
            if (block.isTile()) continue;
            Leaf &leaf = leaves[Coord{block.key.i, block.key.j, block.key.k}];
            for (int c = 0; c < N; c++) {
                for (int b = 0; b < N; b++) {
                    for (int a = 0; a < N; a++) {
                        const int slot = (a << 6) | (b << 3) | c;
                        leaf.values[slot] = block.phi[Block::index(a, b, c)];
                        leaf.set.set(slot);
                    }
                }
            }
        }
        for (size_t n = begin; n < end; n++) {
            const Block &block = mBlocks[order[n]];
            if (block.isTile()) continue;
            for (int c = 0; c < M; c++) {
                for (int b = 0; b < M; b++) {
                    for (int a = 0; a < M; a++) {
                        if (a < N && b < N && c < N) continue;
                        const Coord p = Corner(block.key, a, b, c);
                        Leaf &leaf = leaves[LeafOf(p)];
                        const int slot = SlotOf(p);
                        const float v = block.phi[Block::index(a, b, c)];
                        if (!leaf.set.test(slot)) {
                            leaf.values[slot] = v;
                            leaf.set.set(slot);
                        } else if (std::bit_cast<uint32_t>(leaf.values[slot]) != std::bit_cast<uint32_t>(v)) {
                            parity[n - begin] = 1;
                        }
                    }
                }
            }
        }

        Voxels<float> phi, tiles;
        std::array<Voxels<float>, 8> parityPhi;
        Voxels<int32_t> flags;
        std::vector<Coord> leafOrder;
        leafOrder.reserve(leaves.size());
        for (const auto &entry : leaves) leafOrder.push_back(entry.first);
        std::sort(leafOrder.begin(), leafOrder.end());
        for (const Coord &l : leafOrder) {
            const Leaf &leaf = leaves.at(l);
            for (int slot = 0; slot < kLeafValues; slot++) {
                if (leaf.set.test(slot)) phi.add(Coord{N * l.x + (slot >> 6), N * l.y + ((slot >> 3) & 7), N * l.z + (slot & 7)}, leaf.values[slot]);
            }
        }
        for (size_t n = begin; n < end; n++) {
            const Block &block = mBlocks[order[n]];
            const Coord at{block.key.i, block.key.j, block.key.k};
            int32_t f = kPresent;
            if (block.active) f |= kActive;
            if (inActiveList[order[n]]) f |= kActiveLeaf;
            if (block.isTile()) {
                f |= kTile;
                tiles.add(at, block.tileValue);
            } else if (parity[n - begin]) {
                f |= kParity;
                Voxels<float> &target = parityPhi[Parity(block.key)];
                for (int c = 0; c < M; c++) {
                    for (int b = 0; b < M; b++) {
                        for (int a = 0; a < M; a++) target.add(Corner(block.key, a, b, c), block.phi[Block::index(a, b, c)]);
                    }
                }
            }
            flags.add(at, f);
        }

        if (!phi.coords.empty() && !writer.addFloatGrid(DepthName(prefix, "phi_d", depth), h, phi.coords, phi.values, mBandWidth, true, error)) return false;
        if (!writer.addInt32Grid(DepthName(prefix, "blocks_d", depth), h * N, flags.coords, flags.values, error)) return false;
        if (!tiles.coords.empty() && !writer.addFloatGrid(DepthName(prefix, "tiles_d", depth), h * N, tiles.coords, tiles.values, mBandWidth, false, error))
            return false;
        for (int p = 0; p < 8; p++) {
            if (!parityPhi[p].coords.empty() &&
                !writer.addFloatGrid(ParityName(prefix, depth, p), h, parityPhi[p].coords, parityPhi[p].values, mBandWidth, true, error))
                return false;
        }
        begin = end;
    }

    const std::string settings = DepthName(prefix, "blocks_d", 0);
    return writer.setMetadata(settings, "springls.version", kVdbVersion, error) && writer.setMetadata(settings, "springls.maxDepth", mMaxDepth, error) &&
           writer.setMetadata(settings, "springls.blocksPerAxis", std::array<int32_t, 3>{mBlocksPerAxis[0], mBlocksPerAxis[1], mBlocksPerAxis[2]}, error) &&
           writer.setMetadata(settings, "springls.domainMax", std::array<float, 3>{mDomainMax[0], mDomainMax[1], mDomainMax[2]}, error) &&
           writer.setMetadata(settings, "springls.bandWidth", mBandWidth, error) && writer.setMetadata(settings, "springls.activeCells", mActiveCells, error) &&
           writer.setMetadata(settings, "springls.activeMargin", mActiveMargin, error);
}

bool AdaptiveGrid3D::readVdb(const VdbReader &reader, const std::string &prefix, std::string &error)
{
    const char *kDamaged = "the VDB file does not hold a springls adaptive grid (or is damaged)";
    const auto fail = [&]() {
        error = kDamaged;
        return false;
    };
    const VdbGrid *settings = reader.find(DepthName(prefix, "blocks_d", 0));
    int32_t version = 0, maxDepth = 0;
    std::array<int32_t, 3> bpa{};
    std::array<float, 3> domainMax{};
    float bandWidth = 0, activeCells = 0, activeMargin = 0;
    if (!settings || !settings->metadata("springls.version", version) || !settings->metadata("springls.maxDepth", maxDepth) ||
        !settings->metadata("springls.blocksPerAxis", bpa) || !settings->metadata("springls.domainMax", domainMax) ||
        !settings->metadata("springls.bandWidth", bandWidth) || !settings->metadata("springls.activeCells", activeCells) ||
        !settings->metadata("springls.activeMargin", activeMargin)) {
        return fail();
    }
    if (version != kVdbVersion) {
        error = "the VDB file is of a different springls version";
        return false;
    }
    if (maxDepth < 0 || maxDepth > kMaxSupportedDepth || bpa[0] <= 0 || bpa[1] <= 0 || bpa[2] <= 0) return fail();

    // Blocks, in key order.
    struct Loaded {
        Key key;
        int32_t flags = 0;
    };
    std::vector<Loaded> loaded;
    for (int depth = 0; depth <= maxDepth; depth++) {
        const VdbGrid *grid = reader.find(DepthName(prefix, "blocks_d", depth));
        if (!grid) break;
        if (!grid->isInt32()) return fail();
        grid->forEachInt32([&](const VdbCoord &p, int32_t f) {
            if (f & kPresent) loaded.push_back(Loaded{Key{depth, p.x, p.y, p.z}, f});
        });
    }
    std::sort(loaded.begin(), loaded.end(), [](const Loaded &a, const Loaded &b) { return a.key < b.key; });
    std::unordered_map<Key, int, KeyHash> keyIndex;
    for (size_t n = 0; n < loaded.size(); n++) keyIndex[loaded[n].key] = static_cast<int>(n);
    size_t roots = 0;
    for (const Loaded &l : loaded) {
        if (l.key.depth != 0) continue;
        if (l.key.i < 0 || l.key.j < 0 || l.key.k < 0 || l.key.i >= bpa[0] || l.key.j >= bpa[1] || l.key.k >= bpa[2]) return fail();
        roots++;
    }
    if (roots != static_cast<size_t>(bpa[0]) * bpa[1] * bpa[2]) return fail();

    std::vector<Block> blocks(loaded.size());
    std::vector<int> activeLeaves;
    for (size_t n = 0; n < loaded.size(); n++) {
        const Loaded &l = loaded[n];
        Block &block = blocks[n];
        block.key = l.key;
        block.h = std::ldexp(1.0f, -l.key.depth);
        block.origin = Vec3(static_cast<float>(l.key.i), static_cast<float>(l.key.j), static_cast<float>(l.key.k)) * (static_cast<float>(N) * block.h);
        block.active = (l.flags & kActive) != 0;
        block.alive = true;
        if (l.flags & kActiveLeaf) activeLeaves.push_back(static_cast<int>(n));
        if (l.key.depth > 0) {
            const auto parent = keyIndex.find(Key{l.key.depth - 1, l.key.i >> 1, l.key.j >> 1, l.key.k >> 1});
            if (parent == keyIndex.end()) return fail();
            block.parent = parent->second;
            blocks[parent->second].children[(l.key.i & 1) | ((l.key.j & 1) << 1) | ((l.key.k & 1) << 2)] = static_cast<int>(n);
        }
        bool ok = true;
        if (l.flags & kTile) {
            const VdbGrid *tiles = reader.find(DepthName(prefix, "tiles_d", l.key.depth));
            float v = 0.0f;
            ok = tiles && tiles->get(VdbCoord{l.key.i, l.key.j, l.key.k}, v);
            block.makeTile(v);
        } else {
            const VdbGrid *source = reader.find((l.flags & kParity) ? ParityName(prefix, l.key.depth, Parity(l.key)) : DepthName(prefix, "phi_d", l.key.depth));
            ok = source != nullptr;
            block.materialize();
            for (int c = 0; c < M && ok; c++) {
                for (int b = 0; b < M && ok; b++) {
                    for (int a = 0; a < M && ok; a++) ok = source->get(ToVdb(Corner(l.key, a, b, c)), block.phi[Block::index(a, b, c)]);
                }
            }
        }
        if (!ok) return fail();
    }
    // Activity was listed in leaves() order: every listed block must be a leaf.
    for (int index : activeLeaves) {
        if (!blocks[index].isLeaf()) return fail();
    }

    mBlocks = std::move(blocks);
    mFreeList.clear();
    mActiveLeaves = std::move(activeLeaves);
    for (int a = 0; a < 3; a++) {
        mBlocksPerAxis[a] = bpa[a];
        mDomainMax[a] = domainMax[a];
    }
    mMaxDepth = maxDepth;
    mBandWidth = bandWidth;
    mActiveCells = activeCells;
    mActiveMargin = activeMargin;
    restoreDerived();
    return true;
}

bool AdaptiveGrid3D::saveVdb(std::ostream &out, const std::vector<std::pair<std::string, std::string>> &metadata, std::string &error) const
{
    VdbWriter writer;
    if (!writeVdb(writer, "", error)) return false;
    for (const auto &[name, text] : metadata) {
        if (!writer.setMetadata("blocks_d0", kMetaPrefix + name, text, error)) return false;
    }
    return writer.write(out, error);
}

bool AdaptiveGrid3D::loadVdb(std::istream &in, std::vector<std::pair<std::string, std::string>> &metadata, std::string &error)
{
    VdbReader reader;
    if (!reader.read(in, error)) return false;
    if (!readVdb(reader, "", error)) return false;
    metadata.clear();
    for (const auto &[name, text] : reader.find("blocks_d0")->stringMetadata()) {
        if (name.rfind(kMetaPrefix, 0) == 0) metadata.emplace_back(name.substr(kMetaPrefix.size()), text);
    }
    return true;
}
}  // namespace springls
