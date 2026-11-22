#pragma once
#include <core/Image.hpp>
#include <levelset/MultiNarrowBand2D.hpp>

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
//
// A labeled grid (initializeLabeled) stores the MOGAC representation instead of phi: a region
// label per corner and, in phi, the unsigned distance to the nearest region boundary. Refinement,
// coarsening and synchronization then interpolate each candidate label's signed value (-d inside
// the region, +d outside) and pick a label with the two-smallest rule (MogacUpdate), so they never
// invent a label and never average distances across a boundary. Unlabeled grids are unaffected.
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
        template <class Archive>
        void serialize(Archive &ar)
        {
            ar(depth, i, j);
        }
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
        // Labeled grids: corner labels, empty exactly when phi is (a tile has tileLabel everywhere).
        bool labeled = false;
        std::vector<int> labels;
        int tileLabel = 0;
        int parent = -1;
        std::array<int, 4> children{-1, -1, -1, -1};
        // Leaves around this leaf (-1 outside the domain); valid while the plans are valid.
        std::array<int, kNeighborSlots> neighbors{};
        bool alive = true;
        bool active = false;  // inside the narrow band (evolved)

        // cereal serialization (checkpoints).
        template <class Archive>
        void serialize(Archive &ar)
        {
            ar(key, h, origin, phi, tileValue, labeled, labels, tileLabel, parent, children, neighbors, alive, active);
        }
        bool isLeaf() const { return children[0] < 0; }
        bool isTile() const { return phi.empty(); }
        static int index(int a, int b) { return b * kBlockNodes + a; }
        float value(int node) const { return phi.empty() ? tileValue : phi[node]; }
        int label(int node) const { return labels.empty() ? tileLabel : labels[node]; }
        // Signed value of region l at a corner of a labeled block.
        float signedValue(int node, int l) const { return (label(node) == l) ? -value(node) : value(node); }
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
            if (labeled && labels.empty()) labels.assign(kNodeCount, tileLabel);
        }
        void makeTile(float value)
        {
            tileValue = value;
            std::vector<float>().swap(phi);
            std::vector<int>().swap(labels);
        }
        void makeTile(float value, int l)
        {
            makeTile(value);
            tileLabel = l;
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
    // Labeled grid from a label per pixel and the unsigned distance (pixels) to region boundaries.
    void initializeLabeled(const ImageI &labels, const ImageF &distance, int maxDepth, float bandWidth);
    bool isLabeled() const { return mLabeled; }
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
    // Labeled grids: label and unsigned distance at p (two-smallest rule on the bilinear signed
    // values of the labels at the containing cell's corners).
    MogacResult sampleLabel(const Vec2 &p) const;
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

    // --- Labeled grids (MOGAC). ---
    // evolve() for a labeled grid: each zero-layer corner (distance <= h / 2) may take its own
    // label or a 4-neighbor's; every candidate label's signed values move by upwind advection with
    // `velocities` plus curvature (objects: curvatureWeight; background label 0: backgroundCurvatureWeight,
    // and advection only if advectBackground: springl pulls move objects only, a flow moves every
    // region), and the two-smallest rule picks the label and distance.
    // Distance layers are rebuilt and isolated corners plugged, as in MultiNarrowBand2D.
    void evolveLabeled(int iterations, float curvatureWeight, float backgroundCurvatureWeight, const std::vector<NodeVelocities> &velocities,
                       bool advectBackground = false);
    // Measured correction: measure(position, h, label, d) may report that a corner of an active
    // leaf lies inside region `label` at distance d; a corner with another label takes it. Returns
    // how many corners changed.
    int correctLabels(const std::function<bool(const Vec2 &position, float h, int &label, float &d)> &measure);
    // Labeled contour: extractContour() of each object label's signed values (labels above 0, in
    // ascending order), each label with its own vertexes; curveLabels[c] is the label of curves[c].
    // Where three or more regions meet in a cell, each label's curve passes through the cell's
    // junction vertex (FindJunction on the block's lattice; one copy per label); junctionVertices,
    // when given, flags those (1) among all vertexes (0).
    void extractLabeledContour(Points2 &vertices, Curves &curves, std::vector<int> &curveLabels, std::vector<float> &vertexScales,
                               std::vector<uint8_t> *junctionVertices = nullptr) const;

    size_t leafNodeCount() const { return mLeaves.size() * static_cast<size_t>(kNodeCount); }
    // Approximate heap + object bytes held by the grid (blocks, arrays, index, plans).
    size_t memoryBytes() const;
    size_t tileCount() const;

    // cereal serialization (checkpoints): every block (dead ones too, so indexes and the free list
    // stay the same) and the leaf lists; the key index is rebuilt and the plans on demand.
    template <class Archive>
    void save(Archive &ar) const
    {
        ar(mBlocks, mFreeList, mLeaves, mActiveLeaves, mBlocksX, mBlocksY, mMaxDepth, mBandWidth, mDomainMax, mLabeled);
    }
    template <class Archive>
    void load(Archive &ar)
    {
        ar(mBlocks, mFreeList, mLeaves, mActiveLeaves, mBlocksX, mBlocksY, mMaxDepth, mBandWidth, mDomainMax, mLabeled);
        mIndex.clear();
        for (int n = 0; n < static_cast<int>(mBlocks.size()); n++) {
            if (mBlocks[n].alive) mIndex[mBlocks[n].key] = n;
        }
        invalidatePlans();
    }

protected:
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
    bool mLabeled = false;

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
    // synchronize() of a labeled grid.
    void synchronizeLabeled();
    // gatherLocal() of a labeled grid: unsigned distances and labels with the one-node ring.
    void gatherLocalLabeled(int index, std::array<float, kGhostWidth * kGhostWidth> &distance, std::array<int, kGhostWidth * kGhostWidth> &labels) const;
    // Marching squares on value(block, node) over the leaves (see extractContour()). junction(block,
    // a, b, J), when given, names cells whose single segment u -> v is routed through J (world
    // position) as u -> J -> v, and listed in junctionVertices if given.
    void extractContourOf(const std::function<float(const Block &, int)> &value, bool skipTiles, Points2 &vertices, Curves &curves,
                          std::vector<float> &vertexScales, const std::function<bool(const Block &, int, int, Vec2 &)> &junction = nullptr,
                          std::vector<uint32_t> *junctionVertices = nullptr) const;
    // Turns inactive leaves whose corners are all +band (or all -band) into tiles.
    void compactTiles();
};
}  // namespace springls
