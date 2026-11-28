// IntrinsicTriangulation: signposts over the input mesh, Lawson flips to intrinsic Delaunay,
// correspondence with the input mesh.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic/IntrinsicTriangulation.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <set>
#include <string>

using namespace springls;
using springls_test::TestMesh;

namespace
{
constexpr double kTwoPi = 2.0 * std::numbers::pi;

IntrinsicTriangulation Built(const TestMesh &mesh)
{
    IntrinsicTriangulation tri;
    std::string error;
    CHECK(tri.build(mesh.vertices, mesh.triangles, error));
    if (!error.empty()) std::printf("build: %s\n", error.c_str());
    return tri;
}

double TotalArea(const IntrinsicTriangulation &tri)
{
    double area = 0.0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) area += tri.faceArea(f);
    return area;
}

double Wrap(double a)
{
    a = std::fmod(a, kTwoPi);
    return (a < 0.0) ? a + kTwoPi : a;
}
}  // namespace

TEST(build_rejects_boundary_and_nonmanifold)
{
    IntrinsicTriangulation tri;
    std::string error;
    const Points3 v = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 1, 1)};
    CHECK(!tri.build(v, {{0, 1, 2}}, error));
    CHECK(error.find("boundary") != std::string::npos);
    // A closed tetrahedron plus a third triangle on edge 0-1.
    error.clear();
    CHECK(!tri.build(v, {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 1, 4}}, error));
    CHECK(error.find("non-manifold") != std::string::npos);
    // The tetrahedron alone is fine.
    error.clear();
    CHECK(!tri.build(v, {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}, error));  // vertex 4 unused
    CHECK(error.find("not used") != std::string::npos);
    error.clear();
    CHECK(tri.build(Points3(v.begin(), v.begin() + 4), {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}, error));
    CHECK(tri.vertexCount() == 4);
}

TEST(layout_matches_lengths)
{
    const IntrinsicTriangulation tri = Built(springls_test::Icosphere(3));
    double worst = 0.0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const auto p = tri.layout(f);
        CHECK(p[0].norm() == 0.0 && p[1].y() == 0.0 && p[1].x() > 0.0 && p[2].y() > 0.0);
        for (int c = 0; c < 3; c++) worst = std::max(worst, std::abs((p[(c + 1) % 3] - p[c]).norm() - tri.length(tri.edge(3 * f + c))));
    }
    CHECK(worst < 1e-12);
    CHECK(tri.vertexCount() == 642 && tri.faceCount() == 1280 && tri.edgeCount() == 1920);
}

TEST(signposts_sum_to_full_turn)
{
    IntrinsicTriangulation tri = Built(springls_test::SkewedIcosphere(2));
    tri.flipToDelaunay();
    double worst = 0.0, worstTotal = 0.0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const uint32_t start = tri.vertexHalfedge(v);
        uint32_t he = start;
        double total = 0.0;
        do {
            CHECK(tri.tail(he) == v);
            const double normalized = tri.cornerAngle(he) * kTwoPi / tri.angleSum(v);
            const uint32_t next = tri.nextOutgoing(he);
            worst = std::max(worst, std::abs(Wrap(tri.signpost(next) - tri.signpost(he)) - normalized));
            total += normalized;
            he = next;
        } while (he != start);
        worstTotal = std::max(worstTotal, std::abs(total - kTwoPi));
    }
    CHECK(worst < 1e-9);
    CHECK(worstTotal < 1e-9);
}

TEST(flip_preserves_area_and_vertex_count)
{
    const TestMesh mesh = springls_test::SkewedIcosphere(2);
    IntrinsicTriangulation tri = Built(mesh);
    const double area = TotalArea(tri);
    std::vector<double> lengths(tri.edgeCount());
    for (uint32_t e = 0; e < tri.edgeCount(); e++) lengths[e] = tri.length(e);
    std::vector<std::array<uint32_t, 2>> ends(tri.edgeCount());
    for (uint32_t e = 0; e < tri.edgeCount(); e++) {
        const uint32_t he = tri.edgeHalfedge(e);
        ends[e] = {std::min(tri.tail(he), tri.head(he)), std::max(tri.tail(he), tri.head(he))};
    }
    const size_t flips = tri.flipToDelaunay();
    CHECK(flips > 0);
    CHECK(tri.vertexCount() == mesh.vertices.size());
    CHECK(tri.faceCount() == mesh.triangles.size());
    CHECK_NEAR(TotalArea(tri), area, 1e-9 * area);
    // Edges that were never flipped keep their endpoints and length.
    size_t kept = 0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) {
        const uint32_t he = tri.edgeHalfedge(e);
        const std::array<uint32_t, 2> now = {std::min(tri.tail(he), tri.head(he)), std::max(tri.tail(he), tri.head(he))};
        if (now == ends[e]) {
            kept++;
            CHECK_NEAR(tri.length(e), lengths[e], 1e-12);
        }
    }
    CHECK(kept > tri.edgeCount() / 2);
}

TEST(delaunay_after_flips)
{
    IntrinsicTriangulation tri = Built(springls_test::Icosphere(3));
    tri.flipToDelaunay();
    double worst = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) worst = std::min(worst, tri.cotanWeight(e));
    CHECK(worst >= -1e-12);
}

TEST(delaunay_has_nonnegative_cotans_on_skewed_sphere)
{
    IntrinsicTriangulation tri = Built(springls_test::SkewedIcosphere(3));
    double before = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) before = std::min(before, tri.cotanWeight(e));
    CHECK(before < -0.1);  // the stretched mesh is far from Delaunay
    tri.flipToDelaunay();
    double worst = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) worst = std::min(worst, tri.cotanWeight(e));
    CHECK(worst >= -1e-12);
}

TEST(to_extrinsic_vertex_and_flipped_face)
{
    const TestMesh mesh = springls_test::Icosphere(1);
    IntrinsicTriangulation tri = Built(mesh);
    // Vertex barycentrics map to the vertex.
    for (uint32_t f = 0; f < tri.faceCount(); f += 7) {
        for (int c = 0; c < 3; c++) {
            Vec3d bary = Vec3d::Zero();
            bary[c] = 1.0;
            const auto p = tri.toExtrinsic(f, bary);
            CHECK((p.position - mesh.vertices[tri.tail(3 * f + c)].cast<double>()).norm() < 1e-6);
        }
    }
    // Flip one edge ij (faces ijk and jil) and map the centroid of the new face kil.
    const uint32_t e = 5;
    const uint32_t he = tri.edgeHalfedge(e);
    const uint32_t i = tri.tail(he), j = tri.head(he), k = tri.tail(tri.prev(he)), l = tri.tail(tri.prev(tri.twin(he)));
    CHECK(tri.flipEdge(e));
    uint32_t newFace = UINT32_MAX;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const std::set<uint32_t> corners = {tri.tail(3 * f), tri.tail(3 * f + 1), tri.tail(3 * f + 2)};
        if (corners == std::set<uint32_t>{i, k, l}) newFace = f;
    }
    CHECK(newFace != UINT32_MAX);
    if (newFace == UINT32_MAX) return;
    const auto P = [&](uint32_t v) { return mesh.vertices[v].cast<double>(); };
    // Hinge layout of the two original faces: i at the origin, j on +x, k above, l below.
    const double lij = (P(i) - P(j)).norm();
    const auto place = [&](uint32_t v, double side) {
        const double a = (P(v) - P(i)).norm(), b = (P(v) - P(j)).norm();
        const double x = (a * a + lij * lij - b * b) / (2.0 * lij);
        return Vec2d(x, side * std::sqrt(std::max(0.0, a * a - x * x)));
    };
    const Vec2d pi(0, 0), pj(lij, 0), pk = place(k, 1.0), pl = place(l, -1.0);
    const Vec2d centroid = (pi + pk + pl) / 3.0;
    // Barycentric coordinates of the centroid in the original face it falls in, then 3D.
    const auto bary2 = [](const Vec2d &a, const Vec2d &b, const Vec2d &c, const Vec2d &x) {
        const auto cr = [](const Vec2d &u, const Vec2d &w) { return u.x() * w.y() - u.y() * w.x(); };
        const double A = cr(b - a, c - a);
        return Vec3d(cr(b - x, c - x) / A, cr(c - x, a - x) / A, cr(a - x, b - x) / A);
    };
    Vec3d expected;
    if (centroid.y() >= 0.0) {
        const Vec3d w = bary2(pi, pj, pk, centroid);
        expected = w[0] * P(i) + w[1] * P(j) + w[2] * P(k);
    } else {
        const Vec3d w = bary2(pj, pi, pl, centroid);
        expected = w[0] * P(j) + w[1] * P(i) + w[2] * P(l);
    }
    const auto mapped = tri.toExtrinsic(newFace, Vec3d(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));
    CHECK((mapped.position - expected).norm() < 1e-5);
}

TEST(planar_grid_has_no_flips_needed_for_right_triangles)
{
    IntrinsicTriangulation tri = Built(springls_test::DoubledGrid(8));
    CHECK(tri.flipToDelaunay() == 0);
    // Unit-length sides on both sheets (shared on the border) plus one sqrt(2) diagonal per square per sheet.
    const double sides = 2.0 * (2 * 8 * 9) - 4 * 8, diagonals = 2.0 * 64;
    CHECK(tri.edgeCount() == static_cast<size_t>(sides + diagonals));
    CHECK_NEAR(tri.meanEdgeLength(), (sides + diagonals * std::sqrt(2.0)) / (sides + diagonals), 1e-6);
    // Interior vertices of each sheet are flat.
    double worst = 0.0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        if (tri.degree(v) >= 6) worst = std::max(worst, std::abs(tri.angleSum(v) - kTwoPi));
    }
    CHECK(worst < 1e-9);
}
