// Geodesic fast marching on intrinsic Delaunay triangulations.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic/IntrinsicDistance.hpp>

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

double Arc(const IntrinsicTriangulation &tri, uint32_t a, uint32_t b)
{
    const Vec3d p = tri.inputVertices()[a].cast<double>().normalized(), q = tri.inputVertices()[b].cast<double>().normalized();
    return std::acos(std::clamp(p.dot(q), -1.0, 1.0));
}
}  // namespace

TEST(march_planar_matches_euclidean)
{
    const int n = 30;
    const IntrinsicTriangulation tri = Delaunay(springls_test::DoubledGrid(n));
    const uint32_t center = 15 * (n + 1) + 15;
    const std::vector<double> D = FastMarch(tri, {{center, 0.0}});
    double worst = 0.0;
    for (int j = 0; j <= n; j++) {
        for (int i = 0; i <= n; i++) {
            const double e = std::hypot(i - 15.0, j - 15.0);
            if (e < 3.0 || e > 10.0 * tri.meanEdgeLength()) continue;
            worst = std::max(worst, std::abs(D[j * (n + 1) + i] - e) / e);
        }
    }
    CHECK(worst < 0.02);
}

TEST(march_icosphere_matches_arc_length)
{
    const IntrinsicTriangulation tri = Delaunay(springls_test::Icosphere(4));
    const std::vector<double> D = FastMarch(tri, {{0, 0.0}});
    const double h = tri.meanEdgeLength();
    double worst = 0.0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const double a = Arc(tri, 0, v);
        if (a < 3.0 * h) continue;
        worst = std::max(worst, std::abs(D[v] - a) / a);
    }
    CHECK(worst < 0.02);
}

TEST(march_gradient_unit_length)
{
    const IntrinsicTriangulation tri = Delaunay(springls_test::Icosphere(4));
    const std::vector<double> D = FastMarch(tri, {{0, 0.0}});
    const double h = tri.meanEdgeLength();
    double worst = 0.0;
    size_t counted = 0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        double lo = 1e30, hi = 0.0;
        for (int c = 0; c < 3; c++) {
            lo = std::min(lo, D[tri.tail(3 * f + c)]);
            hi = std::max(hi, D[tri.tail(3 * f + c)]);
        }
        // Away from the seed and the antipode (cut locus), where even exact distances interpolate
        // linearly with a gradient several percent off within ~7 edges.
        if (lo < 6.0 * h || hi > kPi - 6.0 * h) continue;
        worst = std::max(worst, std::abs(FaceGradient(tri, f, D).norm() - 1.0));
        counted++;
    }
    CHECK(counted > tri.faceCount() / 2);
    CHECK(worst < 0.05);
}

TEST(march_respects_max_distance)
{
    const IntrinsicTriangulation tri = Delaunay(springls_test::Icosphere(3));
    const double h = tri.meanEdgeLength();
    const std::vector<double> D = FastMarch(tri, {{0, 0.0}}, 0.5);
    size_t inside = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const double a = Arc(tri, 0, v);
        if (a < 0.45) {
            CHECK(std::isfinite(D[v]));
            inside++;
        }
        if (a > 0.5 + 2.0 * h) CHECK(std::isinf(D[v]));
        if (std::isfinite(D[v])) CHECK(D[v] <= 0.5 + 1.5 * h);
    }
    CHECK(inside > 10);
}

TEST(seeds_from_segment_are_exact_in_face)
{
    const IntrinsicTriangulation tri = Delaunay(springls_test::Icosphere(3));
    const uint32_t f = 42;
    const SurfacePoint a{f, Vec3d(0.6, 0.3, 0.1)}, b{f, Vec3d(0.1, 0.3, 0.6)};
    const std::vector<DistanceSeed> seeds = SeedsFromSegments(tri, {{a, b}});
    CHECK(seeds.size() == 3);
    const auto P = tri.layout(f);
    const Vec2d pa = tri.toLayout(a), pb = tri.toLayout(b);
    for (const DistanceSeed &s : seeds) {
        int corner = -1;
        for (int c = 0; c < 3; c++) {
            if (tri.tail(3 * f + c) == s.vertex) corner = c;
        }
        CHECK(corner >= 0);
        if (corner < 0) continue;
        const Vec2d x = P[corner];
        const double t = std::clamp((x - pa).dot(pb - pa) / (pb - pa).squaredNorm(), 0.0, 1.0);
        CHECK_NEAR(s.distance, (x - (pa + t * (pb - pa))).norm(), 1e-12);
    }
    // Segments in two faces seed both faces' vertices.
    const uint32_t he = 3 * f;
    const SurfacePoint c{tri.face(tri.twin(he)), Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    CHECK(SeedsFromSegments(tri, {{a, c}}).size() == 4);
}

TEST(seeds_are_never_lowered)
{
    const IntrinsicTriangulation tri = Delaunay(springls_test::Icosphere(3));
    const uint32_t he = tri.vertexHalfedge(0);
    const uint32_t neighbor = tri.head(he);
    const std::vector<double> D = FastMarch(tri, {{0, 0.0}, {neighbor, 5.0}});
    CHECK(D[neighbor] == 5.0);
    CHECK(D[0] == 0.0);
    CHECK(D[tri.head(tri.nextOutgoing(he))] < 2.0 * tri.meanEdgeLength());
}

TEST(point_source_neighborhood_yields_to_closer_seeds)
{
    // An isolated seed's exact neighborhood is only a starting value: a closer seed wins.
    const IntrinsicTriangulation tri = Delaunay(springls_test::Icosphere(3));
    const std::vector<double> alone = FastMarch(tri, {{0, 0.0}});
    // A pair of adjacent seeds two rings out, at distance 0.
    const uint32_t he = tri.vertexHalfedge(0);
    const uint32_t ring1 = tri.head(he);
    uint32_t far = UINT32_MAX;
    {
        const uint32_t start = tri.vertexHalfedge(ring1);
        uint32_t h = start;
        do {
            if (alone[tri.head(h)] > 1.5 * tri.meanEdgeLength()) far = tri.head(h);
            h = tri.nextOutgoing(h);
        } while (h != start && far == UINT32_MAX);
    }
    CHECK(far != UINT32_MAX);
    const uint32_t farNext = tri.head(tri.vertexHalfedge(far));
    const std::vector<double> D = FastMarch(tri, {{0, 0.0}, {far, 0.0}, {farNext, 0.0}});
    // ring1 is adjacent to `far`, so it is at most one edge from a seed at 0.
    double edge = 0.0;
    {
        const uint32_t start = tri.vertexHalfedge(ring1);
        uint32_t h = start;
        do {
            if (tri.head(h) == far) edge = tri.halfedgeLength(h);
            h = tri.nextOutgoing(h);
        } while (h != start);
    }
    CHECK(D[ring1] <= std::min(alone[ring1], edge) + 1e-12);
}
