#pragma once
#include <core/Common3D.hpp>
#include <io/Ply.hpp>

namespace springls
{
// Placement of an adaptive grid around a mesh. Grid space is measured in base voxels, with the
// mesh's bounding-box center at the center of the grid.
struct MeshGridFit {
    Vec3 origin = Vec3::Zero();       // mesh-space position of grid corner (0,0,0)
    float baseVoxel = 1.0f;           // mesh units per base voxel
    float finestCell = 1.0f;          // mesh units per finest cell
    Vec3i baseCells = Vec3i::Zero();  // base cells per axis, multiples of AdaptiveGrid3D::kBlockCells
    int maxDepth = 5;                 // refinement depth (finest cell = baseVoxel / 2^maxDepth)
    bool clamped = false;             // finest resolution hit maxFinestCells
    Vec3 toGrid(const Vec3 &m) const { return (m - origin) / baseVoxel; }
    Vec3 toMesh(const Vec3 &g) const { return origin + g * baseVoxel; }
};

struct MeshGridOptions {
    int maxDepth = -1;              // >= 0: explicit depth (clamped to the grid's maximum); -1: ChooseMaxDepth()
    float cellScale = 1.0f;         // finest cell = mean edge length * cellScale
    int gridOverride = 0;           // > 0: base cells along the longest axis
    float padVoxels = 2.0f + 3.0f;  // empty base voxels on each side: 2 + AdaptiveTuning().bandWidth
    int maxFinestCells = 16384;     // cap on finest cells along the longest axis
};

// Adaptive depth for a mesh of triangleCount triangles: round(log2(sqrt(T) / 24)), clamped to
// [1, AdaptiveGrid3D::kMaxSupportedDepth] (about 24 finest cells across a face of a uniform mesh per
// base voxel, so the base grid stays coarse and the octree carries the resolution).
int ChooseMaxDepth(size_t triangleCount);

// Sizes a grid for the mesh from its mean edge length (or gridOverride), centered on its bounds.
MeshGridFit FitMeshGrid(const PolygonMesh &mesh, const MeshGridOptions &options);
}  // namespace springls
