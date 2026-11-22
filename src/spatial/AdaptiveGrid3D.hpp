#pragma once
#include <core/Volume.hpp>
#include <levelset/IsoSurface3D.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace springls
{
class VdbWriter;
class VdbReader;

// Block-adaptive 3D level set grid (sparse octree of uniform blocks); 3D counterpart of
// AdaptiveGrid2D.
//
// Depth 0 tiles the domain with blocks of kBlockCells^3 cells of size 1 (one voxel of the input
// level set). A block at depth d has cells of size 2^-d; refining a leaf replaces it with eight
// children one depth finer. Leaves that touch (faces, edges or corners) differ by at most one
// depth (2:1 balance).
//
// Values (phi, in world units) live at cell corners, so every coarse corner is also a fine
// corner. synchronize() makes the field continuous: at shared corners the finest leaf is
// authoritative, and on a coarse/fine face or edge the fine leaf's extra corners are the linear
// (on edges) or bilinear (inside faces) interpolation of the coarse corners around them. Both
// sides then see the same piecewise-trilinear phi on the border.
//
// Memory: a block stores only its corner values, and blocks entirely outside the narrow band are
// tiles that store a single value. Work buffers are transient and sized to the active blocks.
class AdaptiveGrid3D
{
public:
    static constexpr int kBlockCells = 8;
    static constexpr int kBlockNodes = kBlockCells + 1;
    static constexpr int kNodeCount = kBlockNodes * kBlockNodes * kBlockNodes;
    static constexpr int kCellCount = kBlockCells * kBlockCells * kBlockCells;
    static constexpr int kMaxSupportedDepth = 10;
    // Neighbor slots of a block: four per face (quadrants), two per edge (halves), one per corner.
    static constexpr int kFaceSlots = 6 * 4, kEdgeSlots = 12 * 2, kCornerSlots = 8;
    static constexpr int kNeighborSlots = kFaceSlots + kEdgeSlots + kCornerSlots;

    using NodeVelocities = std::array<Vec3, kNodeCount>;

    struct Key {
        int depth = 0;
        int i = 0, j = 0, k = 0;
        bool operator==(const Key &o) const { return depth == o.depth && i == o.i && j == o.j && k == o.k; }
        template <class Archive>
        void serialize(Archive &ar)
        {
            ar(depth, i, j, k);
        }
        bool operator<(const Key &o) const
        {
            if (depth != o.depth) return depth < o.depth;
            if (k != o.k) return k < o.k;
            if (j != o.j) return j < o.j;
            return i < o.i;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &key) const
        {
            return (static_cast<size_t>(key.depth) * 73856093u) ^ (static_cast<size_t>(key.i) * 19349663u) ^ (static_cast<size_t>(key.j) * 83492791u) ^
                   (static_cast<size_t>(key.k) * 2654435761u);
        }
    };

    struct Block {
        Key key;
        float h = 1.0f;  // cell size (world units)
        Vec3 origin = Vec3::Zero();
        // Corner values (kNodeCount), or empty for a tile whose every corner is tileValue.
        std::vector<float> phi;
        float tileValue = 0.0f;
        int parent = -1;
        std::array<int, 8> children{-1, -1, -1, -1, -1, -1, -1, -1};
        // Leaves around this leaf (-1 outside the domain); valid while the plans are valid.
        std::array<int, kNeighborSlots> neighbors{};
        bool alive = true;
        bool active = false;  // inside the narrow band (evolved)

        bool isLeaf() const { return children[0] < 0; }
        // cereal serialization (checkpoints); neighbors are rebuilt with the plans.
        template <class Archive>
        void serialize(Archive &ar)
        {
            ar(key, h, origin, phi, tileValue, parent, children, alive, active);
        }
        bool isTile() const { return phi.empty(); }
        static int index(int a, int b, int c) { return (c * kBlockNodes + b) * kBlockNodes + a; }
        float value(int node) const { return phi.empty() ? tileValue : phi[node]; }
        float at(int a, int b, int c) const { return value(index(a, b, c)); }
        void materialize()
        {
            if (phi.empty()) phi.assign(kNodeCount, tileValue);
        }
        void makeTile(float v)
        {
            tileValue = v;
            std::vector<float>().swap(phi);
        }
        Vec3 nodePosition(int a, int b, int c) const { return origin + h * Vec3(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)); }
        float size() const { return h * kBlockCells; }
        // Trilinear interpolation at local (cell-unit) coordinates, clamped to the block.
        float interpolate(const Vec3 &u) const;
    };

    // Builds depth-0 blocks covering the corner lattice of `levelSet` (corner (i, j, k) = voxel (i, j, k)).
    void initialize(const VolumeF &levelSet, int maxDepth, float bandWidth);
    void clear();

    int maxDepth() const { return mMaxDepth; }
    float bandWidth() const { return mBandWidth; }
    // Narrow band in local voxels (as in OpenVDB): a leaf with cells of size h is active while
    // phi comes within cells * h + margin of zero (margin in world units, for how far the interface
    // moves between band updates), capped by bandWidth.
    void setActiveBand(float cells, float margin)
    {
        mActiveCells = cells;
        mActiveMargin = margin;
    }
    float activeThreshold(float h) const { return std::min(mBandWidth * (1.0f - 1E-4f), mActiveCells * h + mActiveMargin); }
    Vec3 domainMax() const { return mDomainMax; }
    const std::vector<Block> &blocks() const { return mBlocks; }
    std::vector<Block> &blocks() { return mBlocks; }
    // Live leaf indexes, sorted by key so iteration is deterministic.
    const std::vector<int> &leaves() const { return mLeaves; }
    // Active leaves (inside the narrow band), in leaves() order; set by updateActive().
    const std::vector<int> &activeLeaves() const { return mActiveLeaves; }
    int finestDepth() const;

    // Leaf containing p (p is clamped to the domain).
    int findLeaf(const Vec3 &p) const;
    // Trilinear interpolation of phi in the leaf containing p.
    float sample(const Vec3 &p) const;
    // Cell size of the leaf containing p.
    float cellSize(const Vec3 &p) const { return mBlocks[findLeaf(p)].h; }

    // Splits leaf `index` into eight children (refining neighbors first to keep 2:1 balance).
    // Children inherit phi by trilinear interpolation. Returns false at the maximum depth.
    bool refine(int index);
    // refine() for each of `indexes` that is still a leaf (in order), rebuilding the leaf list once at the
    // end instead of after every split. Returns how many were split.
    int refineLeaves(const std::vector<int> &indexes);
    // Whether the eight children of `index` are leaves and merging them keeps 2:1 balance.
    bool canCoarsen(int index) const;
    // Merges the eight (leaf) children of `index` back into it.
    void coarsen(int index);

    // Enforces continuity (see the class comment).
    void synchronize();
    // Marks leaves whose |phi| reaches below the band as active and lists them.
    void updateActive();
    // Pull velocity (cell units) at a node position, for a leaf with cells of size h.
    using VelocityFunction = std::function<Vec3(const Vec3 &position, float h)>;
    // Runs `iterations` sparse-field steps of phi_t + v . grad(phi) = curvatureWeight * kappa on the
    // active leaves: the zero layer (|phi| <= h / 2) moves by explicit upwinding (at most half a
    // cell per step), the next three layers are rebuilt from it as distances, and isolated
    // zero-layer corners are pushed out. velocity() is evaluated lazily, once per zero-layer corner.
    void evolve(int iterations, float curvatureWeight, const VelocityFunction &velocity);
    // Measured correction: for every corner of an active leaf, measure(position, h, d) may supply a
    // signed distance d; where its sign disagrees with phi, phi takes d (only corners on the wrong
    // side change, as in the particle level set method). Returns how many corners changed.
    int correctSigns(const std::function<bool(const Vec3 &position, float h, float &d)> &measure);
    // Surface nets on the octree: one vertex per crossed cell, and one polygon per minimal crossed
    // cell edge (an edge no finer cell subdivides) joining the smallest cells around it: a quad, or
    // a triangle where one coarser cell appears twice. The mesh is watertight across coarse/fine
    // borders. Faces are wound counterclockwise about the outward normal. Triangle meshes split
    // quads along the shorter diagonal; quad meshes turn triangles into quads with the midpoint of
    // their longest edge. vertexScales[v] is the cell size of vertex v's cell.
    void extractSurface(ElementType type, SurfaceMesh3D &mesh, std::vector<float> &vertexScales);

    size_t tileCount() const;

    // The grid as an OpenVDB file (written and read with tinyvdb), viewable in VDB tools:
    //  - "phi_d<d>": a float level set per depth d, voxel size 2^-d, where voxel (8i + a, 8j + b, 8k + c) is
    //    corner (a, b, c) of block (d, i, j, k), so each block is one VDB leaf node and blocks share their
    //    border corners. Tiles are left out (VDB background).
    //  - "blocks_d<d>" (int32) and "tiles_d<d>" (float) at block coordinates (voxel size 8 * 2^-d): every
    //    live block's flags, and each tile's value.
    //  - "phi_d<d>_p<n>": every corner of the rare blocks whose far-face corners differ from what the
    //    neighbor sharing them stores, split by block parity n so these blocks never touch.
    // Band settings and `metadata` (string pairs, e.g. what the grid is of) are stored on "blocks_d0".
    // Every live block's values and flags round-trip exactly; blocks are renumbered in key order (dead
    // blocks are dropped). Return false with a message on failure; a failed loadVdb() leaves the grid
    // unchanged.
    bool saveVdb(std::ostream &out, const std::vector<std::pair<std::string, std::string>> &metadata, std::string &error) const;
    bool loadVdb(std::istream &in, std::vector<std::pair<std::string, std::string>> &metadata, std::string &error);
    // The same grids added to (or read from) a file shared with others, every grid name prefixed by `prefix`.
    bool writeVdb(VdbWriter &writer, const std::string &prefix, std::string &error) const;
    bool readVdb(const VdbReader &reader, const std::string &prefix, std::string &error);

    // cereal serialization (checkpoints): every block (dead ones too, so indexes stay the same) and
    // the band settings; the key index, leaf list and plans are rebuilt on load.
    template <class Archive>
    void save(Archive &ar) const
    {
        ar(mBlocks, mFreeList, mActiveLeaves, mBlocksPerAxis[0], mBlocksPerAxis[1], mBlocksPerAxis[2], mMaxDepth, mBandWidth, mActiveCells, mActiveMargin,
           mDomainMax);
    }
    template <class Archive>
    void load(Archive &ar)
    {
        ar(mBlocks, mFreeList, mActiveLeaves, mBlocksPerAxis[0], mBlocksPerAxis[1], mBlocksPerAxis[2], mMaxDepth, mBandWidth, mActiveCells, mActiveMargin,
           mDomainMax);
        restoreDerived();
    }

private:
    struct SyncEntry {
        int32_t block;
        uint16_t node;
        uint8_t depth;
    };
    struct HangingRule {
        int32_t block;
        uint16_t node;
        uint8_t count;  // 2 (edge midpoint) or 4 (face center)
        std::array<uint16_t, 4> sources;
    };
    static constexpr int kGhostWidth = kBlockNodes + 2;  // local array with a one-node ring
    static constexpr int kGhostCount = kGhostWidth * kGhostWidth * kGhostWidth;
    using Local = std::array<float, kGhostCount>;

    std::vector<Block> mBlocks;
    std::unordered_map<Key, int, KeyHash> mIndex;
    std::vector<int> mFreeList;
    std::vector<int> mLeaves;
    std::vector<int> mActiveLeaves;
    int mBlocksPerAxis[3] = {0, 0, 0};
    int mMaxDepth = 0;
    float mBandWidth = 3.0f;
    float mActiveCells = 3.0f, mActiveMargin = 0.5f;
    Vec3 mDomainMax = Vec3::Zero();

    // Topology-dependent plans, rebuilt lazily after refine / coarsen / initialize.
    bool mPlansValid = false;
    bool mDeferLeafList = false;       // inside refineLeaves()
    std::vector<int> mSyncGroupStart;  // groups of corners sharing one position
    std::vector<SyncEntry> mSyncEntries;
    std::vector<HangingRule> mHangingRules;

    int allocate(const Block &block);
    void release(int index);
    void rebuildLeafList();
    // After load(): rebuilds the key index and leaf list, and invalidates the plans.
    void restoreDerived();
    void invalidatePlans() { mPlansValid = false; }
    void buildPlans();
    bool inDomain(const Vec3 &p) const { return (p.array() >= 0.0f).all() && (p.array() <= mDomainMax.array()).all(); }
    // Leaf containing local position u (cell units of leaf `index`, possibly outside the block),
    // through the neighbor slots: `index` itself inside the block, -1 outside the domain.
    int leafAt(int index, const Vec3 &u) const;
    void gatherLocal(int index, Local &local) const;
    void compactTiles();
};
}  // namespace springls
