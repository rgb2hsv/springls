#pragma once
// Checks of a meshlet level's isometric UVs (MeshletLevel::texcoords): flips, overlaps, angles, areas, and
// how far apart corresponding corners are.
#include <lod/MeshletHierarchy.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace springls
{
// Two corners that are the same mesh vertex in two triangles of one region that share an edge: corner
// cornerA of triangle faceA and corner cornerB of triangle faceB. A corner has at most two correspondences,
// one across each of its triangle's two edges at it, and only to triangles of the same region.
struct MeshletCornerPair {
    uint32_t faceA, faceB;
    int cornerA, cornerB;
};
std::vector<MeshletCornerPair> MeshletCornerPairs(const MeshletLevel &level);

// Overlaps are measured in texels of a texture of this size covering the unit UV square: a pair counts as
// overlapping only when its interiors overlap by more than one texel.
constexpr int kOverlapCheckTextureSize = 4096;

struct MeshletUvChecks {
    size_t triangles = 0;
    size_t flips = 0;         // UV triangles with non-positive signed area
    size_t overlaps = 0;             // pairs of UV triangles overlapping by more than one texel (any regions)
    size_t crossRegionOverlaps = 0;  // ... of them from different regions (charts running into each other)
    double maxOverlapTexels = 0.0;   // deepest overlap of any pair, in texels
    std::vector<uint32_t> overlappingTriangles;  // triangles in at least one of those pairs, sorted
    double maxAngleError = 0.0;  // largest |UV angle - 3D angle|, radians
    // UV area / 3D area of every triangle, divided by its region's median: 1 everywhere for an authalic map.
    double minAreaRatio = 1.0, maxAreaRatio = 1.0;
    // Distance between corresponding corners (MeshletCornerPairs), UV units.
    double meanCornerGap = 0.0, maxCornerGap = 0.0;
};
MeshletUvChecks CheckMeshletUvs(const MeshletLevel &level);
}  // namespace springls
