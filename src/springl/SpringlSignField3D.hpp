#pragma once
#include <spatial/AdaptiveGrid3D.hpp>
#include <spatial/TriangleTree3D.hpp>

#include <cstdint>
#include <functional>
#include <vector>

namespace springls
{
struct SignFieldParams3D {
    float capsuleCells = 1.5f;  // capsule radius r = max(capsuleCells * h, kMinBandRadius * h, minBandVoxels), h the finest cell
    // Absolute floor on the capsule radius (base voxels): springls split after a refinement keep the
    // gaps they had before the split until fill closes them.
    float minBandVoxels = 0.3f;
    // Lattice nodes store their distance to the springls out to this far (and phi() clamps there).
    float distanceClamp = 1.0f;
};

struct SignFieldInput3D {
    const AdaptiveGrid3D *grid = nullptr;  // the grid being rebuilt (its leaves, before the rebuild)
    const Points3 *particles = nullptr;
    const Points3 *vertexes = nullptr;
    const std::vector<uint32_t> *start = nullptr;  // springl n owns vertexes [start[n], start[n + 1])
    std::vector<uint32_t> springls;                // the springls that define this field
};

// Level set reconstructed from 3D springls (SemiLagrangianDweaveLevelSet3D). As in the
// original spring level sets, the springls define only an unsigned distance; inside and outside
// come from topology, not from springl normals:
//
// 1. U is the min distance to the springls' fan triangles (the distance to the union of their
//    elements). Their capsules, the nodes with U <= r, are walls.
// 2. The exterior floods in from the domain border around the walls; non-wall nodes it does not
//    reach are inside (so an outside pocket sealed off by walls is filled).
// 3. The sign propagates upwind into the walls: wall nodes in order of decreasing U, each taking
//    the sign of its neighbors with larger U weighted by how much larger, so the outside and
//    inside fronts meet at U = 0, on the springls. A node with no upwind neighbor (a local maximum
//    of U inside a wall: the middle of a sheet thinner than two capsules) is inside, so thin sheets
//    survive; the price is that fronts closer than two capsules fuse (a union).
// 4. phi = sign * min(U, distanceClamp).
//
// The field is sampled on a sparse lattice: the nodes of the finest leaves (the grid's own finest
// nodes), with every coarser leaf one node of the flood's graph. That is exact: the walls (within
// about 3 finest cells of a particle) lie inside the finest leaves, which reach
// AdaptiveSpringLevelSet3D's band margin around every particle, so coarse leaves never touch a
// wall and each lies wholly inside or outside.
//
// With no springls at all there is no surface: outside everywhere. The input pointers must outlive
// the field and stay unchanged while it is sampled.
class SpringlSignField3D
{
public:
    // Smallest capsule radius, in finest cells: a wall thinner than half a cell could pass between
    // lattice nodes and let the flood leak through.
    static constexpr float kMinBandRadius = 0.75f;

    void build(const SignFieldInput3D &input, const SignFieldParams3D &params);
    // Finest cell size (lattice spacing) and capsule radius in use.
    float latticeSpacing() const { return mH; }
    float bandRadius() const { return mBandRadius; }
    // U(p): min distance to the springl triangles, at most maxDistance (stored at lattice nodes, up
    // to distanceClamp).
    float distance(const Vec3 &p, float maxDistance) const;
    // +1 outside, -1 inside, at a node of the grid passed to build(): a lattice node, or any node
    // of a coarse leaf.
    float sign(const Vec3 &p) const;
    float phi(const Vec3 &p, float clamp) const { return sign(p) * distance(p, clamp); }
    // Whether some springl lies within `distance` of p.
    bool anyWithin(const Vec3 &p, float distance) const { return mTree.nearest(p, distance).has_value(); }

    // Diagnostics.
    size_t latticeNodeCount() const { return mPositions.size(); }
    size_t wallNodeCount() const { return mWallCount; }

private:
    // Lattice node at p (rounded), or -1.
    int latticeAt(const Vec3 &p) const;
    bool isCoarseLeaf(int leaf) const { return leaf >= 0 && mCoarseNode[leaf] >= 0; }
    // Graph of lattice nodes (neighbors[6 n + axis]: a lattice node, a coarse leaf's graph node
    // >= L, or -1) and coarse leaves (joined across faces), and the nodes the exterior reaches from
    // the domain border without crossing a wall.
    void floodExterior(const std::vector<int> &coarseLeaves, std::vector<int> &neighbors, std::vector<uint8_t> &exterior) const;

    SignFieldInput3D mInput;
    SignFieldParams3D mParams;
    TriangleTree3D mTree;
    float mH = 1.0f, mBandRadius = 1.0f;
    // Lattice: a dense index over the finest leaves' bounding box, node positions, and per node
    // whether it is a wall, its sign and min(U, distanceClamp).
    int64_t mLatticeLo[3] = {0, 0, 0};  // lattice coordinates of the dense index's corner
    Vec3i mLatticeDims = Vec3i::Zero();
    std::vector<int> mLatticeIndex;  // lattice node of each dense slot, or -1
    Points3 mPositions;
    std::vector<uint8_t> mWall;
    std::vector<int8_t> mSign;
    std::vector<float> mDistance;
    // Per grid block index: its node in the flood's graph (lattice count + n) for coarse leaves, or
    // -1; and the sign of each coarse leaf.
    std::vector<int> mCoarseNode;
    std::vector<int8_t> mCoarseSign;
    size_t mWallCount = 0;
};

// The grid's level set rebuilt from a field (semi-Lagrangian springls, mesh tracking): phi = sign * min(U, clamp) at
// the nodes of the finest leaves the surface comes within clamp of; every other leaf becomes a tile of +-band with
// its side's sign (coarser leaves never hold the surface). Leaves whose nodes all sit at the clamp on one side become
// tiles too. The caller synchronizes the grid and updates its active set.
void FillGridFromSignField(AdaptiveGrid3D &grid, const SpringlSignField3D &field, float band, float clamp);
}  // namespace springls
