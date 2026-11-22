#pragma once
// Geodesic distance on an intrinsic triangulation by fast marching [Kimmel and Sethian, "Computing
// geodesic paths on manifolds", PNAS 1998; Sethian 1999], on intrinsic edge lengths and halfedge
// walks: a corrected port of wedge's MeshDistanceField (planar-front triangle update, one-level
// unfolding across the opposite edge for obtuse corners, Dijkstra-like edge update otherwise).
#include <intrinsic/IntrinsicTriangulation.hpp>

#include <limits>
#include <utility>
#include <vector>

namespace springls
{
struct DistanceSeed {
    uint32_t vertex;
    double distance;
};
// Fast marching from fixed seeds (never lowered); stops beyond maxDistance (vertices left at +inf).
// Seeds none of which is adjacent to another are point sources (circular fronts, exact near each
// source); otherwise the seeds sample a front. Deterministic (ties by vertex index); double precision.
std::vector<double> FastMarch(const IntrinsicTriangulation &tri, const std::vector<DistanceSeed> &seeds,
                              double maxDistance = std::numeric_limits<double>::infinity());
// Seeds from segments on the surface: every vertex of a face a segment's endpoint lies in gets its
// exact in-layout distance to the segment (minimum over segments; endpoints in different faces are
// related by LocalLog). One seed per vertex, ascending vertex order.
std::vector<DistanceSeed> SeedsFromSegments(const IntrinsicTriangulation &tri, const std::vector<std::pair<SurfacePoint, SurfacePoint>> &segments);
// Gradient of the piecewise-linear interpolant of vertex values, constant on the face, in layout(face).
Vec2d FaceGradient(const IntrinsicTriangulation &tri, uint32_t face, const std::vector<double> &vertexValues);
}  // namespace springls
