#pragma once
// Labeled regions on intrinsic test meshes: a label per vertex from a function of its 3D position,
// and the unsigned geodesic distance to the region boundaries (fast marching from the boundary's
// segments, which cross each face between the edge points where the labels change).
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic/IntrinsicDistance.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace springls_test
{
inline springls::IntrinsicTriangulation DelaunaySurface(const TestMesh &mesh)
{
    springls::IntrinsicTriangulation tri;
    std::string error;
    tri.build(mesh.vertices, mesh.triangles, error);
    tri.flipToDelaunay();
    return tri;
}

struct Regions {
    std::vector<int> labels;
    std::vector<double> distance;
};

// label(p) picks each vertex's region; level(p) is a function whose sign change places the boundary
// on an edge between two labels (crossing by linear interpolation of |level|).
inline Regions MakeRegions(const springls::IntrinsicTriangulation &tri, const std::function<int(const springls::Vec3d &)> &label,
                           const std::function<double(const springls::Vec3d &)> &level)
{
    using namespace springls;
    Regions r;
    const size_t V = tri.vertexCount();
    r.labels.resize(V);
    std::vector<double> value(V);
    for (uint32_t v = 0; v < V; v++) {
        const Vec3d p = tri.vertexPosition(v);
        r.labels[v] = label(p);
        value[v] = std::abs(level(p));
    }
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        std::vector<SurfacePoint> crossings;
        for (int c = 0; c < 3; c++) {
            const uint32_t he = 3 * f + c;
            const uint32_t a = tri.tail(he), b = tri.head(he);
            if (r.labels[a] == r.labels[b]) continue;
            const double t = value[a] / std::max(1e-300, value[a] + value[b]);
            crossings.push_back(IntrinsicTriangulation::halfedgePoint(he, t));
        }
        if (crossings.size() == 2) segments.emplace_back(crossings[0], crossings[1]);
        if (crossings.size() == 3) {
            const SurfacePoint center{f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
            for (const SurfacePoint &p : crossings) segments.emplace_back(p, center);
        }
    }
    r.distance = FastMarch(tri, SeedsFromSegments(tri, segments));
    return r;
}

// Tangent field on every face: the projection of a fixed 3D direction onto the face's plane, in the
// face's layout, scaled to `length`.
inline std::vector<springls::Vec2d> ProjectedField(const springls::IntrinsicTriangulation &tri, const springls::Vec3d &direction, double length)
{
    using namespace springls;
    std::vector<Vec2d> field(tri.faceCount(), Vec2d::Zero());
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const Vec3d P0 = tri.vertexPosition(tri.tail(3 * f));
        const Vec3d P1 = tri.vertexPosition(tri.tail(3 * f + 1));
        const Vec3d P2 = tri.vertexPosition(tri.tail(3 * f + 2));
        const auto L = tri.layout(f);
        Eigen::Matrix<double, 3, 2> E;
        E << (P1 - P0), (P2 - P0);
        // Least-squares coefficients of the direction in the face's edge basis.
        const Eigen::Vector2d w = (E.transpose() * E).ldlt().solve(E.transpose() * direction);
        const Vec2d v = w[0] * (L[1] - L[0]) + w[1] * (L[2] - L[0]);
        if (v.norm() > 0.0) field[f] = length * v.normalized();
    }
    return field;
}
// A 3D tangent vector at a point of `face` expressed in the face's layout (least squares in the
// face's edge basis; exact for vectors in the plane of the face's corners).
inline springls::Vec2d TangentToLayout(const springls::IntrinsicTriangulation &tri, uint32_t face, const springls::Vec3d &v)
{
    using namespace springls;
    const Vec3d P0 = tri.vertexPosition(tri.tail(3 * face));
    const Vec3d P1 = tri.vertexPosition(tri.tail(3 * face + 1));
    const Vec3d P2 = tri.vertexPosition(tri.tail(3 * face + 2));
    const auto L = tri.layout(face);
    Eigen::Matrix<double, 3, 2> E;
    E << (P1 - P0), (P2 - P0);
    const Eigen::Vector2d w = (E.transpose() * E).ldlt().solve(E.transpose() * v);
    return w[0] * (L[1] - L[0]) + w[1] * (L[2] - L[0]);
}

// 3D position of a surface point from its face's corners (exact on unflipped, unrefined faces).
inline springls::Vec3d CornerPosition(const springls::IntrinsicTriangulation &tri, const springls::SurfacePoint &p)
{
    springls::Vec3d x = springls::Vec3d::Zero();
    for (int c = 0; c < 3; c++) x += p.bary[c] * tri.vertexPosition(tri.tail(3 * p.face + c));
    return x;
}

// Star R(θ) = r (1 + 0.35 cos 5θ) on the unit sphere around `center`, in the tangent frame (east, north)
// at the center; split by its local north axis: label 1 west half, label 2 east half, 0 outside.
inline Regions StarRegions(const springls::IntrinsicTriangulation &tri, const springls::Vec3d &center, double r)
{
    using namespace springls;
    const Vec3d c = center.normalized();
    const Vec3d east = Vec3d(0, 0, 1).cross(c).normalized(), north = c.cross(east);
    const auto local = [=](const Vec3d &p) {
        const Vec3d q = p.normalized();
        const double angle = std::acos(std::clamp(q.dot(c), -1.0, 1.0));
        const Vec3d t = q - q.dot(c) * c;
        const double n = t.norm();
        const Vec2d dir = (n > 0.0) ? Vec2d(t.dot(east) / n, t.dot(north) / n) : Vec2d(1, 0);
        return Vec2d(angle * dir);  // azimuthal equidistant coordinates
    };
    const auto outline = [=](const Vec3d &p) {
        const Vec2d x = local(p);
        const double theta = std::atan2(x.y(), x.x());
        return x.norm() - r * (1.0 + 0.35 * std::cos(5.0 * theta));
    };
    const auto label = [=](const Vec3d &p) { return outline(p) >= 0.0 ? 0 : (local(p).x() < 0.0 ? 1 : 2); };
    const auto level = [=](const Vec3d &p) {
        const double o = std::abs(outline(p));
        return (outline(p) >= 0.0) ? o : std::min(o, std::abs(local(p).x()));
    };
    return MakeRegions(tri, label, level);
}

// One region: negative inside (-d), positive outside, d the distance to the region's boundary.
inline std::vector<double> SignedPhi(const Regions &r)
{
    std::vector<double> phi(r.labels.size());
    for (size_t v = 0; v < phi.size(); v++) phi[v] = (r.labels[v] > 0) ? -r.distance[v] : r.distance[v];
    return phi;
}

// The star R(θ) = r (1 + amplitude cos 5θ) as one region (no split), signed: SignedPhi of its regions.
inline std::vector<double> StarPhi(const springls::IntrinsicTriangulation &tri, const springls::Vec3d &center, double r, double amplitude = 0.35)
{
    using namespace springls;
    const Vec3d c = center.normalized();
    const Vec3d east = Vec3d(0, 0, 1).cross(c).normalized(), north = c.cross(east);
    const auto outline = [=](const Vec3d &p) {
        const Vec3d q = p.normalized();
        const double angle = std::acos(std::clamp(q.dot(c), -1.0, 1.0));
        const Vec3d t = q - q.dot(c) * c;
        const double n = t.norm();
        const double theta = (n > 0.0) ? std::atan2(t.dot(north), t.dot(east)) : 0.0;
        return angle - r * (1.0 + amplitude * std::cos(5.0 * theta));
    };
    return SignedPhi(MakeRegions(tri, [=](const Vec3d &p) { return outline(p) < 0.0 ? 1 : 0; }, outline));
}
}  // namespace springls_test
