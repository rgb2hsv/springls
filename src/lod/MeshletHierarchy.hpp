#pragma once
// The meshlet level-of-detail hierarchy of a closed triangle mesh, built by meshlet-optimizer
// (external/meshlet-optimizer, MeshletBuilder) and returned as plain springls arrays. Only
// src/lod/mopt/MeshletHierarchy.cpp sees meshlet-optimizer itself.
#include <core/Common3D.hpp>
#include <mesh/LoopSubdivision.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace springls
{
// One level of the hierarchy. Level 0 is the input mesh with the base charts as regions; level k > 0 is
// level k-1 remeshed inside the level-k regions (meshlet-optimizer's LodLevel).
struct MeshletLevel {
    Points3 vertexes;
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<int> faceRegion;    // region (hierarchy node at this level) of each triangle
    std::vector<int> vertexRegion;  // region of each vertex, -1 on a region border
    std::vector<int> regionParent;  // RegionHierarchy::parent of this level: region -> region at the next level; empty at the top
    // Isometric UVs in double precision, 3 per triangle (its corners in order): every triangle has the angles
    // of its 3D triangle, inscribed in its atlas triangle at its region's median scale, then moved and scaled
    // (no change of shape) so that corresponding corners of a region come together and areas become
    // proportional to 3D areas (meshlet-optimizer's "Isometric UVs" stage).
    std::vector<Vec2d> texcoords;
    // The packed atlas the isometric UVs start from (meshlet-optimizer's LSCM charts), 3 per triangle: each region's
    // chart in it bounds the region's isometric UVs when uvFitToAtlas is on.
    std::vector<Vec2d> atlasTexcoords;
};

struct MeshletHierarchyParams {
    int maxDepth = 5;               // clustering passes above the base (MeshletParams::maxDepth)
    uint32_t randomSeed = 3429841;  // MeshletParams::randomSeed
    int uvRelaxSteps = 400;           // rigid-body steps that pull the isometric UV triangles together (MeshletParams::uvRelaxSteps)
    bool uvEnforceNoOverlaps = true;  // UV triangles repel each other until none overlap
    bool uvFitToAtlas = true;         // then scale (uniformly), turn and move each region to fit its atlas chart
};

// How deep the interiors of two UV triangles overlap (UV units; 0 when they do not): meshlet-optimizer's
// separating axis test, the one its isometric UVs are freed of overlaps with.
double UvOverlapDepth(const std::array<Vec2d, 3> &p, const std::array<Vec2d, 3> &q);

// Builds the hierarchy of a closed triangle mesh with meshlet-optimizer's defaults except maxDepth and
// randomSeed (no .mlod file). project, if set, snaps resampled vertexes onto the true surface
// (MeshletBuilder::projectToSurface). Level 0 first. Empty on failure.
std::vector<MeshletLevel> BuildMeshletHierarchy(const LoopMesh &mesh, const MeshletHierarchyParams &params,
                                                const std::function<Vec3(const Vec3 &)> &project = {});
}  // namespace springls
