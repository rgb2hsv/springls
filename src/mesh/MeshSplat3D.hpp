#pragma once
#include <mesh/MeshDistance3D.hpp>
#include <spatial/AdaptiveGrid3D.hpp>

#include <cstddef>

namespace springls
{
struct MeshSplatStats {
    size_t refinedLeaves = 0;     // leaves at maxDepth
    size_t candidateEntries = 0;  // candidate faces summed over those leaves
    size_t fallbackCorners = 0;   // corners resolved by the full query (only when E <= half a leaf diagonal)
    size_t floodComponents = 0;   // face-connected components of the coarser leaves
};

// Initializes `grid` (already initialize()d) from a closed mesh, splatting each face's bounding box into the
// octree instead of sampling at the base voxels. `mesh` is in grid space (base voxels). Refines every leaf
// within r0 of a face to maxDepth, writes signed distances to their corners (exact within E = max(r0, the grid's
// active threshold for a leaf one depth coarser), +-bandWidth beyond), and signs every other leaf by flood fill:
//   1. Refine, one depth at a time: faces are binned to the leaves their r0-grown bounds overlap, and a leaf
//      any of them comes within r0 of (MeshDistance3D::faceNearBox) is split.
//   2. Each maxDepth leaf collects the faces within E of it (binned by their E-grown bounds), cut to those
//      within |phi(center)| + half its diagonal, which still holds every face closest to a corner. A corner
//      takes the signed distance to the closest of them within E; farther corners (and every corner of a leaf
//      whose center is at least E + half its diagonal away, which becomes a tile) take sign(center) *
//      bandWidth. Only beyond E the evolution needs nothing but the sign: it rebuilds every node past the zero
//      layer from its neighbors, and E covers the active band. No full-mesh query runs per corner (only one per
//      leaf center), unless E is below a finest leaf's half diagonal, where such corners need it for the sign
//      (counted in fallbackCorners).
//   3. The other leaves (no face within r0) form face-connected components, each wholly inside or outside.
//      A component takes the majority sign of the corners on its border with maxDepth leaves (with no such
//      border: positive if it touches the domain boundary, else negative), and its leaves become tiles of
//      +-bandWidth. Enclosed cavities therefore come out right, unlike a flood from the domain boundary.
//   4. synchronize() and updateActive().
MeshSplatStats SplatMeshIntoGrid(AdaptiveGrid3D &grid, const MeshDistance3D &mesh, float r0, float bandWidth);
}  // namespace springls
