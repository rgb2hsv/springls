// Discrete exponential map, parallel transport and the local log map on intrinsic triangulations.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic/IntrinsicTrace.hpp>

#include <cmath>
#include <string>

using namespace springls;

namespace
{
IntrinsicTriangulation Delaunay(const springls_test::TestMesh &mesh)
{
    IntrinsicTriangulation tri;
    std::string error;
    CHECK(tri.build(mesh.vertices, mesh.triangles, error));
    tri.flipToDelaunay();
    return tri;
}

Vec3d Position(const IntrinsicTriangulation &tri, const SurfacePoint &p) { return tri.toExtrinsic(p).position; }

// A point of the top sheet of a DoubledGrid(n) at grid position x (away from the border), and the
// linear map from the grid's xy plane to the layout of the face it lies in.
struct GridPoint {
    SurfacePoint point;
    Eigen::Matrix2d toLayout;
};
GridPoint TopPoint(const IntrinsicTriangulation &tri, int n, const Vec2d &x)
{
    const int i = static_cast<int>(x.x()), j = static_cast<int>(x.y());
    for (uint32_t f : {static_cast<uint32_t>(4 * (j * n + i)), static_cast<uint32_t>(4 * (j * n + i) + 1)}) {
        const auto P = [&](int c) { return tri.inputVertices()[tri.tail(3 * f + c)].cast<double>().head<2>().eval(); };
        Eigen::Matrix3d M;
        M << P(0).x(), P(1).x(), P(2).x(), P(0).y(), P(1).y(), P(2).y(), 1, 1, 1;
        const Vec3d bary = M.inverse() * Vec3d(x.x(), x.y(), 1.0);
        if (bary.minCoeff() < -1e-12) continue;
        const auto L = tri.layout(f);
        Eigen::Matrix2d world, local;
        world << (P(1) - P(0)), (P(2) - P(0));
        local << (L[1] - L[0]), (L[2] - L[0]);
        return {SurfacePoint{f, bary}, local * world.inverse()};
    }
    CHECK(false);
    return {};
}
}  // namespace

TEST(trace_planar_equals_translation)
{
    const int n = 20;
    const IntrinsicTriangulation tri = Delaunay(springls_test::DoubledGrid(n));
    const GridPoint g = TopPoint(tri, n, Vec2d(10.3, 10.6));
    const Vec3d s3 = Position(tri, g.point);
    for (const Vec2d &world : {Vec2d(3.3, 2.1), Vec2d(-4.7, 0.6), Vec2d(0.25, -6.0)}) {
        const TraceResult r = Trace(tri, g.point, g.toLayout * world);
        CHECK((Position(tri, r.end) - (s3 + Vec3d(world.x(), world.y(), 0.0))).norm() < 1e-9);
        // Flat: transporting the end direction back gives the start direction.
        CHECK((TransportBack(r, r.direction) - (g.toLayout * world).normalized()).norm() < 1e-9);
    }
}

TEST(trace_through_vertex_continues)
{
    // Flat vertex: the path from (9.4, 9.7) through grid vertex (10, 10) stays straight.
    const int n = 20;
    const IntrinsicTriangulation grid = Delaunay(springls_test::DoubledGrid(n));
    const GridPoint g = TopPoint(grid, n, Vec2d(9.4, 9.7));
    const TraceResult r = Trace(grid, g.point, g.toLayout * Vec2d(1.2, 0.6));
    CHECK((Position(grid, r.end) - Vec3d(10.6, 10.3, 0.0)).norm() < 1e-9);

    // Curved vertex (icosphere): aim at a vertex; the trace passes it and keeps going at the same
    // speed along a near-great circle.
    const IntrinsicTriangulation sphere = Delaunay(springls_test::Icosphere(3));
    const double h = sphere.meanEdgeLength();
    const uint32_t f = 100;
    const SurfacePoint p{f, Vec3d(0.2, 0.4, 0.4)};
    const Vec2d target = sphere.layout(f)[0];
    const Vec2d dir = (target - sphere.toLayout(p)).normalized();
    const double L = (target - sphere.toLayout(p)).norm() + 2.0 * h;
    const TraceResult t = Trace(sphere, p, L * dir);
    const Vec3d a = Position(sphere, p).normalized(), b = Position(sphere, t.end).normalized();
    CHECK(std::abs(std::acos(a.dot(b)) - L) < 0.05 * L);
    // Not stuck at the vertex.
    CHECK((Position(sphere, t.end) - sphere.inputVertices()[sphere.tail(3 * f)].cast<double>()).norm() > 1.5 * h);
}

TEST(trace_closed_geodesic_on_icosphere_returns)
{
    // Exactly closed: around the flat two-sided sheet, a straight line comes back after 2n.
    const int n = 8;
    const IntrinsicTriangulation grid = Delaunay(springls_test::DoubledGrid(n));
    const GridPoint g = TopPoint(grid, n, Vec2d(4.3, 3.45));
    const SurfacePoint start = g.point;
    const TraceResult loop = Trace(grid, start, g.toLayout * Vec2d(0.0, 2.0 * n));
    CHECK((Position(grid, loop.end) - Position(grid, start)).norm() < 0.02 * grid.meanEdgeLength());
    // Icosphere: a geodesic of length 2πR comes back near its start (polyhedral geodesics do not
    // close exactly; over ~60 edges the gap stays under one edge).
    const IntrinsicTriangulation sphere = Delaunay(springls_test::Icosphere(3));
    const double h = sphere.meanEdgeLength();
    const SurfacePoint p{37, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    for (double angle : {0.0, 1.0, 2.0}) {
        const TraceResult r = Trace(sphere, p, 2.0 * kPi * Vec2d(std::cos(angle), std::sin(angle)));
        CHECK((Position(sphere, r.end) - Position(sphere, p)).norm() < 1.0 * h);
    }
}

TEST(transport_round_trip)
{
    const IntrinsicTriangulation sphere = Delaunay(springls_test::SkewedIcosphere(3, 2.0));
    const double h = sphere.meanEdgeLength();
    for (uint32_t f : {3u, 250u, 901u}) {
        const SurfacePoint p{f, Vec3d(0.5, 0.3, 0.2)};
        const Vec2d v = 6.0 * h * Vec2d(0.6, -0.8);
        const TraceResult r = Trace(sphere, p, v);
        const TraceResult back = Trace(sphere, r.end, -v.norm() * r.direction);
        CHECK((Position(sphere, back.end) - Position(sphere, p)).norm() < 1e-9);
        CHECK((TransportBack(r, r.direction) - v.normalized()).norm() < 1e-9);
        CHECK((TransportBack(back, back.direction) + r.direction).norm() < 1e-9);
    }
}

TEST(surface_point_on_edge_is_unique)
{
    const IntrinsicTriangulation sphere = Delaunay(springls_test::Icosphere(3));
    for (uint32_t he : {7u, 400u, 3000u}) {
        const SurfacePoint a = IntrinsicTriangulation::halfedgePoint(he, 0.3);
        const SurfacePoint b = IntrinsicTriangulation::halfedgePoint(sphere.twin(he), 0.7);
        const auto log = LocalLog(sphere, a, b);
        CHECK(log.has_value());
        if (log) CHECK(log->norm() < 1e-12);
        CHECK(GeodesicDistanceLocal(sphere, b, a) < 1e-12);
        CHECK((Position(sphere, a) - Position(sphere, b)).norm() < 1e-9);
    }
}

TEST(local_log_matches_trace)
{
    const IntrinsicTriangulation sphere = Delaunay(springls_test::SkewedIcosphere(3, 1.5));
    const double h = sphere.meanEdgeLength();
    double worst = 0.0;
    for (uint32_t f = 0; f < sphere.faceCount(); f += 97) {
        const SurfacePoint p{f, Vec3d(0.25, 0.25, 0.5)};
        for (int k = 0; k < 6; k++) {
            const double angle = 0.9 * k + 0.1, length = h * (0.3 + 0.3 * k);
            const Vec2d v = length * Vec2d(std::cos(angle), std::sin(angle));
            const TraceResult r = Trace(sphere, p, v);
            const auto log = LocalLog(sphere, p, r.end);
            CHECK(log.has_value());
            if (log) worst = std::max(worst, (*log - v).norm());
        }
    }
    CHECK(worst < 1e-6 * h);
}

TEST(local_log_unreached_returns_nullopt)
{
    const IntrinsicTriangulation sphere = Delaunay(springls_test::Icosphere(3));
    const SurfacePoint p = sphere.vertexPoint(0);
    const TraceResult far = Trace(sphere, p, Vec2d(kPi * 0.9, 0.0));
    CHECK(!LocalLog(sphere, p, far.end, 1).has_value());
    CHECK(std::isinf(GeodesicDistanceLocal(sphere, p, far.end, 1)));
}
