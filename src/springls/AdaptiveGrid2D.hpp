#pragma once
#include <springls/Image.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace springls
{
// Block-adaptive 2D level set grid (sparse quadtree of uniform blocks).
//
// Depth 0 tiles the whole domain with blocks of kBlockCells x kBlockCells cells of size 1 (one
// pixel of the input level set). A block at depth d has cells of size 2^-d; refining a leaf
// replaces it with four children one depth finer. Leaves that touch (edges or corners) differ by
// at most one depth (2:1 balance).
//
// Values (phi, in world units) live at cell corners, so every coarse corner is also a fine
// corner. synchronize() makes the field continuous and crack-free: at shared corners the finest
// leaf is authoritative, and along a coarse/fine border the fine leaf's extra corners are the
// linear interpolation of their neighbors. Both sides then see the same piecewise-linear phi on
// the border, so marching squares produces matching crossings.
//
// Memory: a block stores only its corner values, and blocks entirely outside the narrow band
// (every corner at +band or every corner at -band) are tiles that store a single value. Tiles
// turn back into arrays automatically when a shared corner or a refinement needs it. Work buffers
// (velocities, next values) are transient and sized to the active blocks only.
class AdaptiveGrid2D
{
public:
    static constexpr int kBlockCells = 8;
    static constexpr int kBlockNodes = kBlockCells + 1;
    static constexpr int kNodeCount = kBlockNodes * kBlockNodes;
    static constexpr int kMaxSupportedDepth = 12;
    // Neighbor slots of a block: two per side (lower / upper half of the side), then four corners.
    static constexpr int kNeighborSlots = 12;

    using NodeVelocities = std::array<Vec2, kNodeCount>;

    struct Key {
        int depth = 0;
        int i = 0;
        int j = 0;
        bool operator==(const Key &other) const { return depth == other.depth && i == other.i && j == other.j; }
        bool operator<(const Key &other) const
        {
            if (depth != other.depth) return depth < other.depth;
            if (j != other.j) return j < other.j;
            return i < other.i;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const
        {
            return (static_cast<size_t>(k.depth) * 73856093u) ^ (static_cast<size_t>(k.i) * 19349663u) ^ (static_cast<size_t>(k.j) * 83492791u);
        }
    };

    struct Block {
        Key key;
        float h = 1.0f;  // cell size (world units)
        Vec2 origin;     // world position of corner (0, 0)
        // Corner values (kNodeCount), or empty for a tile whose every corner is tileValue.
        std::vector<float> phi;
        float tileValue = 0.0f;
        int parent = -1;
        std::array<int, 4> children{-1, -1, -1, -1};
        // Leaves around this leaf (-1 outside the domain); valid while the plans are valid.
        std::array<int, kNeighborSlots> neighbors{};
        bool alive = true;
        bool active = false;  // inside the narrow band (evolved)

        bool isLeaf() const { return children[0] < 0; }
        bool isTile() const { return phi.empty(); }
        static int index(int a, int b) { return b * kBlockNodes + a; }
        float value(int node) const { return phi.empty() ? tileValue : phi[node]; }
        float at(int a, int b) const { return value(index(a, b)); }
        // Writable access turns a tile into an array first.
        float &at(int a, int b)
        {
            materialize();
            return phi[index(a, b)];
        }
        void materialize()
        {
            if (phi.empty()) phi.assign(kNodeCount, tileValue);
        }
        void makeTile(float value)
        {
            tileValue = value;
            std::vector<float>().swap(phi);
        }
        Vec2 nodePosition(int a, int b) const { return origin + h * Vec2(static_cast<float>(a), static_cast<float>(b)); }
        float size() const { return h * kBlockCells; }
        bool contains(const Vec2 &p) const
        {
            return p.x() >= origin.x() && p.y() >= origin.y() && p.x() <= origin.x() + size() && p.y() <= origin.y() + size();
        }
    };

    // Builds depth-0 blocks covering the corner lattice of `levelSet` (corner (i, j) = pixel (i, j)).
    void initialize(const ImageF &levelSet, int maxDepth, float bandWidth);
    void clear();

    int maxDepth() const { return mMaxDepth; }
    float bandWidth() const { return mBandWidth; }
    Vec2 domainMax() const { return mDomainMax; }
    const std::vector<Block> &blocks() const { return mBlocks; }
    std::vector<Block> &blocks() { return mBlocks; }
    // Live leaf indexes, sorted by key so iteration is deterministic.
    const std::vector<int> &leaves() const { return mLeaves; }
    // Active leaves (inside the narrow band), in leaves() order; set by updateActive().
    const std::vector<int> &activeLeaves() const { return mActiveLeaves; }

    // Leaf containing p (p is clamped to the domain).
    int findLeaf(const Vec2 &p) const;
    // Bilinear interpolation of phi in the leaf containing p.
    float sample(const Vec2 &p) const;
    // Cell size of the leaf containing p.
    float cellSize(const Vec2 &p) const { return mBlocks[findLeaf(p)].h; }

    // Splits leaf `index` into four children (refining neighbors first to keep 2:1 balance).
    // Children inherit phi by bilinear interpolation. Returns false at the maximum depth.
    bool refine(int index);
    // Whether the four children of `index` are leaves and merging them keeps 2:1 balance.
    bool canCoarsen(int index) const;
    // Merges the four (leaf) children of `index` back into it.
    void coarsen(int index);

    // Enforces continuity: shared corners take the finest leaf's value; fine corners on a
    // coarse/fine border are the average of their neighbors along the border.
    void synchronize();
    // Marks leaves whose |phi| reaches below the band as active and lists them.
    void updateActive();

    // Runs `iterations` sparse-field steps of phi_t + v . grad(phi) = curvatureWeight * kappa on
    // the active leaves, with `velocities[n]` (cell units) for activeLeaves()[n]: the zero layer
    // (|phi| <= h / 2) moves by explicit upwinding (at most half a cell per step), the next three
    // layers are rebuilt from it as distances, and isolated zero-layer corners are pushed out
    // (as AdaptiveGrid3D::evolve and NarrowBand2D do). phi is clamped to the band.
    void evolve(int iterations, float curvatureWeight, const std::vector<NodeVelocities> &velocities);

    // Measured correction: for every corner of an active leaf, measure(position, h, d) may supply a
    // signed distance d; where its sign disagrees with phi, phi takes d (only corners on the wrong
    // side change, as in the particle level set method). Returns how many corners changed.
    int correctSigns(const std::function<bool(const Vec2 &position, float h, float &d)> &measure);

    // Marching squares on the leaves; crossings are welded across blocks and chained into curves.
    // Closed curves repeat their first index at the end (same format and winding as IsoContour
    // with Winding::Clockwise). vertexScales[v] is the cell size where vertex v was found.
    void extractContour(Points2 &vertices, Curves &curves, std::vector<float> &vertexScales) const;

    size_t leafNodeCount() const { return mLeaves.size() * static_cast<size_t>(kNodeCount); }
    // Approximate heap + object bytes held by the grid (blocks, arrays, index, plans).
    size_t memoryBytes() const;
    size_t tileCount() const;

private:
    struct SyncEntry {
        int32_t block;
        uint8_t node;
        uint8_t depth;
    };
    struct MidpointRule {
        int32_t block;
        uint8_t node;
        uint8_t nodeA;
        uint8_t nodeB;
    };
    static constexpr int kGhostWidth = kBlockNodes + 2;  // local array with a one-node ring

    std::vector<Block> mBlocks;
    std::unordered_map<Key, int, KeyHash> mIndex;
    std::vector<int> mFreeList;
    std::vector<int> mLeaves;
    std::vector<int> mActiveLeaves;
    int mBlocksX = 0;
    int mBlocksY = 0;
    int mMaxDepth = 0;
    float mBandWidth = 3.0f;
    Vec2 mDomainMax;

    // Topology-dependent plans, rebuilt lazily after refine / coarsen / initialize.
    bool mPlansValid = false;
    std::vector<int> mSyncGroupStart;  // groups of corners sharing one position
    std::vector<SyncEntry> mSyncEntries;
    std::vector<MidpointRule> mMidpointRules;

    int allocate(const Block &block);
    void release(int index);
    void rebuildLeafList();
    void invalidatePlans() { mPlansValid = false; }
    void buildPlans();
    // Fills `local` (kGhostWidth^2, ring included) for leaf `index` from the current phi.
    void gatherLocal(int index, std::array<float, kGhostWidth * kGhostWidth> &local) const;
    // Turns inactive leaves whose corners are all +band (or all -band) into tiles.
    void compactTiles();
};
}  // namespace springls
