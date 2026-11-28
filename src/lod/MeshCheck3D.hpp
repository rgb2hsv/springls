#pragma once
// Checks a closed triangle mesh for fold-overs and overlaps:
// - a fold-over is an edge whose two triangles' normals turn more than foldDegrees apart (the surface doubles back
//   on itself there);
// - an overlap is two triangles that share no vertex and intersect (separating axes), found through a uniform grid
//   of their bounding boxes, in parallel.
// Edges with other than two triangles are counted too.
#include <lod/TriangleSurface3D.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace springls
{
struct MeshCheckParams {
    float foldDegrees = 120.0f;  // adjacent triangle normals further apart than this are a fold-over
};

struct MeshCheck {
    enum Flag : uint8_t { Folded = 1, Overlapping = 2 };
    int foldedEdges = 0;
    int overlappingPairs = 0;
    int nonManifoldEdges = 0;
    std::vector<uint8_t> flags;  // per triangle: Folded | Overlapping
    bool clean() const { return foldedEdges == 0 && overlappingPairs == 0 && nonManifoldEdges == 0; }
};

MeshCheck CheckMesh(const TriangleSurface &mesh, const MeshCheckParams &params = {});

// The pairs of triangles (f < g, sorted) that share no vertex and intersect (separating axes; touching counts).
std::vector<std::pair<uint32_t, uint32_t>> FindIntersectingPairs(const TriangleSurface &mesh);

// Whether two triangles intersect (or touch).
bool TrianglesIntersect(const Vec3 &a0, const Vec3 &a1, const Vec3 &a2, const Vec3 &b0, const Vec3 &b1, const Vec3 &b2);
}  // namespace springls
