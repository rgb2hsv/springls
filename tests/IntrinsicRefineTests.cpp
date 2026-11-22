// IntrinsicTriangulation adaptive refinement: edge splits at intrinsic midpoints, Loop 1:4 splits
// closed by bisection, Delaunay flips, and exact relocation of surface points through all of them.
#include "Check.hpp"
#include "IntrinsicTestMeshes.hpp"

#include <intrinsic/IntrinsicTriangulation.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace springls;
using springls_test::TestMesh;

namespace
{
constexpr double kTwoPi = 6.283185307179586;

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

// The invariants `build` establishes: twins pair up across one edge, every edge has two faces, every
// vertex one fan, positive lengths satisfying the triangle inequality, signposts consistent with the
// corner angles and angle sums.
bool Manifold(const IntrinsicTriangulation &tri)
{
    bool ok = true;
    const size_t H = 3 * tri.faceCount();
    std::vector<int> outgoing(tri.vertexCount(), 0), edgeUses(tri.edgeCount(), 0);
    for (uint32_t he = 0; he < H; he++) {
        const uint32_t tw = tri.twin(he);
        ok = ok && tw < H && tw != he && tri.twin(tw) == he && tri.edge(tw) == tri.edge(he);
        ok = ok && tri.tail(tw) == tri.head(he) && tri.head(tw) == tri.tail(he);
        ok = ok && tri.edge(he) < tri.edgeCount();
        if (tri.edge(he) < tri.edgeCount()) edgeUses[tri.edge(he)]++;
        if (tri.tail(he) < tri.vertexCount()) outgoing[tri.tail(he)]++;
    }
    for (uint32_t e = 0; e < tri.edgeCount(); e++) {
        ok = ok && edgeUses[e] == 2 && tri.edge(tri.edgeHalfedge(e)) == e && tri.length(e) > 0.0;
    }
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const double a = tri.halfedgeLength(3 * f), b = tri.halfedgeLength(3 * f + 1), c = tri.halfedgeLength(3 * f + 2);
        ok = ok && a < b + c && b < c + a && c < a + b;
    }
    double worst = 0.0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        ok = ok && tri.tail(tri.vertexHalfedge(v)) == v && static_cast<int>(tri.degree(v)) == outgoing[v];
        const uint32_t start = tri.vertexHalfedge(v);
        uint32_t he = start;
        double total = 0.0;
        do {
            const double normalized = tri.cornerAngle(he) * kTwoPi / tri.angleSum(v);
            const uint32_t next = tri.nextOutgoing(he);
            if (tri.degree(v) > 1) worst = std::max(worst, std::abs(Wrap(tri.signpost(next) - tri.signpost(he)) - normalized));
            total += tri.cornerAngle(he);
            he = next;
        } while (he != start);
        ok = ok && std::abs(total - tri.angleSum(v)) < 1e-9;
    }
    if (worst > 1e-9) std::printf("signpost mismatch %g\n", worst);
    return ok && worst <= 1e-9;
}

// Random interior barycentrics.
Vec3d RandomBary(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> u(0.01, 1.0);
    Vec3d b(u(rng), u(rng), u(rng));
    return b / b.sum();
}

// Faces sharing an edge with a marked face.
std::vector<uint8_t> Neighbors(const IntrinsicTriangulation &tri, const std::vector<uint8_t> &marks)
{
    std::vector<uint8_t> out(tri.faceCount(), 0);
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        if (!marks[f]) continue;
        for (int c = 0; c < 3; c++) out[tri.face(tri.twin(3 * f + c))] = 1;
    }
    return out;
}

// Geodesic distance on the unit sphere from x to the circle at polar angle `rho` around `pole`.
double CircleDistance(const Vec3d &x, const Vec3d &pole, double rho)
{
    const double angle = std::acos(std::clamp(x.normalized().dot(pole), -1.0, 1.0));
    return std::abs(angle - rho);
}
}  // namespace

TEST(split_edge_keeps_area_and_angle_sums)
{
    const TestMesh mesh = springls_test::Icosphere(2);
    IntrinsicTriangulation tri = Built(mesh);
    const double area = TotalArea(tri);
    const size_t V = tri.vertexCount();
    std::vector<double> sums(V);
    for (uint32_t v = 0; v < V; v++) sums[v] = tri.angleSum(v);
    for (uint32_t e : {0u, 17u, 101u, 200u}) {
        const uint32_t he = tri.edgeHalfedge(e);
        const uint32_t i = tri.tail(he), j = tri.head(he);
        const bool input = i < V && j < V;
        const Vec3d expected = 0.5 * (tri.vertexPosition(i) + tri.vertexPosition(j));
        const size_t F = tri.faceCount(), E = tri.edgeCount();
        const uint32_t m = tri.splitEdge(e);
        CHECK(m == tri.vertexCount() - 1);
        CHECK(tri.faceCount() == F + 2 && tri.edgeCount() == E + 3);
        CHECK_NEAR(tri.angleSum(m), kTwoPi, 1e-15);
        CHECK(tri.degree(m) == 4);
        // On the unflipped input mesh the midpoint of an input edge is its 3D midpoint.
        if (input) CHECK((tri.vertexPosition(m) - expected).norm() < 1e-9);
        CHECK(tri.faceLevel(static_cast<uint32_t>(tri.faceCount() - 1)) == 0);
    }
    CHECK_NEAR(TotalArea(tri), area, 1e-12);
    for (uint32_t v = 0; v < V; v++) CHECK_NEAR(tri.angleSum(v), sums[v], 0.0);
    CHECK(Manifold(tri));
    // Input vertices keep their positions; the vertex point of an inserted vertex maps to its position.
    for (uint32_t v = 0; v < V; v += 13) CHECK((tri.vertexPosition(v) - mesh.vertices[v].cast<double>()).norm() == 0.0);
    for (uint32_t v = static_cast<uint32_t>(V); v < tri.vertexCount(); v++) {
        CHECK((tri.toExtrinsic(tri.vertexPoint(v)).position - tri.vertexPosition(v)).norm() < 1e-9);
    }
}

TEST(split_edge_relocates_points)
{
    IntrinsicTriangulation tri = Built(springls_test::SkewedIcosphere(2));
    tri.flipToDelaunay();
    std::mt19937 rng(3);
    const uint32_t e = 40;
    const uint32_t he = tri.edgeHalfedge(e);
    std::vector<SurfacePoint> points;
    for (uint32_t f : {tri.face(he), tri.face(tri.twin(he)), 7u}) {
        for (int n = 0; n < 20; n++) points.push_back({f, RandomBary(rng)});
    }
    points.push_back(IntrinsicTriangulation::halfedgePoint(he, 0.3));
    points.push_back(IntrinsicTriangulation::halfedgePoint(he, 0.7));
    std::vector<Vec3d> before;
    std::vector<SurfacePoint *> ptrs;
    for (auto &p : points) {
        before.push_back(tri.toExtrinsic(p).position);
        ptrs.push_back(&p);
    }
    tri.splitEdge(e, ptrs);
    double worst = 0.0;
    for (size_t n = 0; n < points.size(); n++) worst = std::max(worst, (tri.toExtrinsic(points[n]).position - before[n]).norm());
    CHECK(worst < 1e-9);
    if (worst >= 1e-9) std::printf("split_edge_relocates_points: worst %g\n", worst);
}

TEST(flip_edge_relocates_points)
{
    IntrinsicTriangulation tri = Built(springls_test::SkewedIcosphere(2));
    std::mt19937 rng(5);
    std::vector<SurfacePoint> points;
    for (uint32_t f = 0; f < tri.faceCount(); f++) points.push_back({f, RandomBary(rng)});
    for (uint32_t he = 0; he < 3 * tri.faceCount(); he += 11) points.push_back(IntrinsicTriangulation::halfedgePoint(he, 0.4));
    std::vector<Vec3d> before;
    std::vector<SurfacePoint *> ptrs;
    for (auto &p : points) {
        before.push_back(tri.toExtrinsic(p).position);
        ptrs.push_back(&p);
    }
    CHECK(tri.flipToDelaunay(ptrs) > 0);
    double worst = 0.0;
    for (size_t n = 0; n < points.size(); n++) worst = std::max(worst, (tri.toExtrinsic(points[n]).position - before[n]).norm());
    CHECK(worst < 1e-9);
    if (worst >= 1e-9) std::printf("flip_edge_relocates_points: worst %g\n", worst);
}

TEST(refine_relocates_points)
{
    for (const bool skewed : {false, true}) {
        IntrinsicTriangulation tri = Built(skewed ? springls_test::SkewedIcosphere(2) : springls_test::Icosphere(2));
        tri.flipToDelaunay();
        std::mt19937 rng(11);
        std::vector<SurfacePoint> points;
        double worst = 0.0;
        for (int pass = 0; pass < 3; pass++) {
            std::vector<uint8_t> marks(tri.faceCount(), 0);
            for (uint32_t f = 0; f < tri.faceCount(); f++) {
                const Vec3d c = tri.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position;
                marks[f] = (c.z() > 0.4 && c.y() > 0.0) ? 1 : 0;
            }
            const std::vector<uint8_t> near = Neighbors(tri, marks);
            for (uint32_t f = 0; f < tri.faceCount(); f++) {
                if (near[f]) points.push_back({f, RandomBary(rng)});
            }
            for (uint32_t f = 0; f < tri.faceCount(); f += 17) points.push_back({f, RandomBary(rng)});
            std::vector<Vec3d> before;
            std::vector<SurfacePoint *> ptrs;
            for (auto &p : points) {
                before.push_back(tri.toExtrinsic(p).position);
                ptrs.push_back(&p);
            }
            const size_t split = tri.refine(marks, ptrs);
            CHECK(split >= static_cast<size_t>(std::count(marks.begin(), marks.end(), 1)));
            for (size_t n = 0; n < points.size(); n++) {
                CHECK(points[n].face < tri.faceCount());
                worst = std::max(worst, (tri.toExtrinsic(points[n]).position - before[n]).norm());
            }
            CHECK(Manifold(tri));
        }
        CHECK(worst < 1e-9);
        if (worst >= 1e-9) std::printf("refine_relocates_points (skewed %d): worst %g\n", skewed ? 1 : 0, worst);
    }
}

TEST(refine_relocates_points_on_split_edges)
{
    IntrinsicTriangulation tri = Built(springls_test::Icosphere(2));
    std::vector<uint8_t> marks(tri.faceCount(), 0);
    for (uint32_t f = 0; f < tri.faceCount(); f += 9) marks[f] = 1;
    std::vector<SurfacePoint> points;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        if (!marks[f]) continue;
        for (int c = 0; c < 3; c++) {
            const uint32_t he = 3 * f + c;
            for (double t : {0.0, 0.25, 0.5, 0.8}) points.push_back(IntrinsicTriangulation::halfedgePoint(he, t));
            // The same edge seen from the neighbor.
            points.push_back(IntrinsicTriangulation::halfedgePoint(tri.twin(he), 0.35));
        }
    }
    std::vector<Vec3d> before;
    std::vector<SurfacePoint *> ptrs;
    for (auto &p : points) {
        before.push_back(tri.toExtrinsic(p).position);
        ptrs.push_back(&p);
    }
    tri.refine(marks, ptrs);
    double worst = 0.0;
    size_t onEdge = 0;
    for (size_t n = 0; n < points.size(); n++) {
        worst = std::max(worst, (tri.toExtrinsic(points[n]).position - before[n]).norm());
        const Vec3d &b = points[n].bary;
        CHECK(b.minCoeff() >= 0.0 && std::abs(b.sum() - 1.0) < 1e-15);
        if (b.minCoeff() == 0.0) onEdge++;
    }
    CHECK(worst < 1e-9);
    if (worst >= 1e-9) std::printf("refine_relocates_points_on_split_edges: worst %g\n", worst);
    CHECK(onEdge == points.size());  // points on split edges stay exactly on edges
}

TEST(refine_near_star_reaches_target)
{
    const TestMesh mesh = springls_test::Icosphere(3);
    IntrinsicTriangulation tri = Built(mesh);
    const IntrinsicTriangulation original = tri;
    double h = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) h = std::max(h, tri.length(e));
    const Vec3d pole = Vec3d(0.3, -0.2, 1.0).normalized();
    const double rho = 0.35;
    const auto distance = [&](const IntrinsicTriangulation &t, uint32_t v) { return CircleDistance(t.vertexPosition(v), pole, rho); };
    const auto faceDistance = [&](const IntrinsicTriangulation &t, uint32_t f) {
        double d = 1e300;
        for (int c = 0; c < 3; c++) d = std::min(d, distance(t, t.tail(3 * f + c)));
        return d;
    };
    for (int pass = 0; pass < 3; pass++) {
        const double target = 1.01 * h / std::pow(2.0, pass + 1);
        std::vector<uint8_t> marks(tri.faceCount(), 0);
        for (uint32_t f = 0; f < tri.faceCount(); f++) {
            double longest = 0.0;
            for (int c = 0; c < 3; c++) longest = std::max(longest, tri.halfedgeLength(3 * f + c));
            marks[f] = (faceDistance(tri, f) < 2.0 * h && longest > target) ? 1 : 0;
        }
        CHECK(tri.refine(marks, {}) > 0);
    }
    CHECK(Manifold(tri));
    size_t near = 0;
    double longestNear = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) {
        const uint32_t he = tri.edgeHalfedge(e);
        if (distance(tri, tri.tail(he)) < 0.5 * h && distance(tri, tri.head(he)) < 0.5 * h) {
            near++;
            longestNear = std::max(longestNear, tri.length(e));
        }
    }
    CHECK(near > 100);
    CHECK(longestNear <= 1.01 * h / 8.0);
    int deepest = 0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) deepest = std::max(deepest, tri.faceLevel(f));
    CHECK(deepest == 3);
    // Far from the circle nothing changed: same number of faces with the same edge lengths.
    const auto farFaces = [&](const IntrinsicTriangulation &t) {
        std::vector<std::array<double, 3>> out;
        for (uint32_t f = 0; f < t.faceCount(); f++) {
            if (faceDistance(t, f) < 4.0 * h) continue;
            std::array<double, 3> l = {t.halfedgeLength(3 * f), t.halfedgeLength(3 * f + 1), t.halfedgeLength(3 * f + 2)};
            std::sort(l.begin(), l.end());
            out.push_back(l);
            CHECK(t.faceLevel(f) == 0);
        }
        std::sort(out.begin(), out.end());
        return out;
    };
    const auto a = farFaces(original), b = farFaces(tri);
    CHECK(a.size() > 100 && a.size() == b.size());
    bool same = a.size() == b.size();
    for (size_t n = 0; same && n < a.size(); n++) {
        for (int c = 0; c < 3; c++) same = same && std::abs(a[n][c] - b[n][c]) < 1e-14;
    }
    CHECK(same);
}

TEST(refine_repeated_levels_stay_manifold)
{
    for (const bool skewed : {false, true}) {
        IntrinsicTriangulation tri = Built(skewed ? springls_test::SkewedIcosphere(2) : springls_test::Icosphere(2));
        tri.flipToDelaunay();
        const Vec3d center = Vec3d(0.2, 0.5, 0.8).normalized();
        for (int pass = 0; pass < 3; pass++) {
            std::vector<uint8_t> marks(tri.faceCount(), 0);
            for (uint32_t f = 0; f < tri.faceCount(); f++) {
                const Vec3d c = tri.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position;
                const Vec3d unit(c.x() / (skewed ? 5.0 : 1.0), c.y(), c.z());
                marks[f] = (unit.normalized().dot(center) > 0.9) ? 1 : 0;
            }
            CHECK(tri.refine(marks, {}) > 0);
            CHECK(Manifold(tri));
            // Delaunay, except the edges of bisected (green) faces, which keep their bisection.
            double worst = 0.0;
            for (uint32_t e = 0; e < tri.edgeCount(); e++) {
                const uint32_t he = tri.edgeHalfedge(e);
                if (tri.bisected(tri.face(he)) || tri.bisected(tri.face(tri.twin(he)))) continue;
                worst = std::min(worst, tri.cotanWeight(e));
            }
            CHECK(worst >= -1e-12);
        }
        int deepest = 0;
        for (uint32_t f = 0; f < tri.faceCount(); f++) deepest = std::max(deepest, tri.faceLevel(f));
        CHECK(deepest == 3);
        // Every vertex maps to its cached position.
        double worst = 0.0;
        for (uint32_t v = 0; v < tri.vertexCount(); v++) worst = std::max(worst, (tri.toExtrinsic(tri.vertexPoint(v)).position - tri.vertexPosition(v)).norm());
        CHECK(worst < 1e-9);
    }
}

TEST(edge_aligned_traces_on_skinny_corner)
{
    // A tetrahedron whose base face has a corner of about 0.005 rad at vertex 0.
    const double t = 0.005;
    const Points3 V = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, static_cast<float>(t), 0), Vec3(0.5f, static_cast<float>(t / 3), 0.3f)};
    const std::vector<std::array<uint32_t, 3>> T = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
    IntrinsicTriangulation tri;
    std::string error;
    CHECK(tri.build(V, T, error));
    double smallest = 10.0;
    for (uint32_t he = 0; he < 3 * tri.faceCount(); he++) smallest = std::min(smallest, tri.cornerAngle(he));
    CHECK(smallest < 0.01);
    const auto P = [&](uint32_t v) { return V[v].cast<double>(); };
    const auto lerp = [&](uint32_t he, double u) { return (1.0 - u) * P(tri.tail(he)) + u * P(tri.head(he)); };
    const auto position = [&](const SurfacePoint &p) {
        Vec3d x = Vec3d::Zero();
        for (int c = 0; c < 3; c++) x += p.bary[c] * P(tri.tail(3 * p.face + c));
        return x;
    };
    double worstTrace = 0.0, worstExtrinsic = 0.0;
    for (uint32_t he = 0; he < 3 * tri.faceCount(); he++) {
        const double L = tri.halfedgeLength(he);
        for (double u : {0.1, 0.5, 0.9}) {
            const auto r = tri.traceFromVertex(tri.tail(he), tri.signpost(he), u * L);
            worstTrace = std::max(worstTrace, (position(r.end) - lerp(he, u)).norm());
            worstExtrinsic = std::max(worstExtrinsic, (tri.toExtrinsic(IntrinsicTriangulation::halfedgePoint(he, u)).position - lerp(he, u)).norm());
        }
    }
    CHECK(worstTrace < 1e-9);
    CHECK(worstExtrinsic < 1e-9);
    if (worstTrace >= 1e-9 || worstExtrinsic >= 1e-9) std::printf("skinny: trace %g extrinsic %g\n", worstTrace, worstExtrinsic);
    // Two levels of splits on every edge: the midpoint, then the midpoints of both halves (each traced
    // from the inserted vertex or the input endpoint).
    double worstSplit = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) {
        IntrinsicTriangulation copy = tri;
        const uint32_t he = copy.edgeHalfedge(e);
        const Vec3d a = P(copy.tail(he)), b = P(copy.head(he));
        const uint32_t m = copy.splitEdge(e);
        worstSplit = std::max(worstSplit, (copy.vertexPosition(m) - 0.5 * (a + b)).norm());
        // splitEdge's documented edges: e is tail-half (a–m), edgeCount()-3 the head-half (m–b).
        const uint32_t headHalf = static_cast<uint32_t>(copy.edgeCount() - 3);
        const uint32_t m2 = copy.splitEdge(headHalf);
        worstSplit = std::max(worstSplit, (copy.vertexPosition(m2) - (0.25 * a + 0.75 * b)).norm());
        const uint32_t m3 = copy.splitEdge(e);
        worstSplit = std::max(worstSplit, (copy.vertexPosition(m3) - (0.75 * a + 0.25 * b)).norm());
        for (uint32_t v : {m, m2, m3}) worstSplit = std::max(worstSplit, (copy.toExtrinsic(copy.vertexPoint(v)).position - copy.vertexPosition(v)).norm());
    }
    CHECK(worstSplit < 1e-9);
    if (worstSplit >= 1e-9) std::printf("skinny: split %g\n", worstSplit);
}

TEST(extrinsic_to_layout_through_correspondence)
{
    // A 3D vector at p, carried into p's intrinsic face layout, must point the way the surface does:
    // with v = Q - X for a point Q in the input face the frame names (X: p's 3D position), tracing p
    // along the result ends at Q exactly (the segment XQ is straight on the input face). An
    // out-of-plane component is projected away. Points in flipped faces (skewed icosphere, Delaunay),
    // in refined faces, at input and inserted vertices and on edges.
    for (const bool skewed : {false, true}) {
        IntrinsicTriangulation tri = Built(skewed ? springls_test::SkewedIcosphere(2) : springls_test::Icosphere(2));
        const size_t flips = tri.flipToDelaunay();
        if (skewed) CHECK(flips > 0);
        std::vector<uint8_t> marks(tri.faceCount(), 0);
        for (uint32_t f = 0; f < tri.faceCount(); f++) marks[f] = tri.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position.z() > 0.3 ? 1 : 0;
        tri.refine(marks, {});
        std::vector<uint8_t> again(tri.faceCount(), 0);
        for (uint32_t f = 0; f < tri.faceCount(); f++) again[f] = tri.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position.z() > 0.6 ? 1 : 0;
        tri.refine(again, {});
        CHECK(tri.vertexCount() > tri.inputVertexCount());
        std::mt19937 rng(5);
        std::vector<SurfacePoint> points;
        for (uint32_t f = 0; f < tri.faceCount(); f++) {
            points.push_back({f, RandomBary(rng)});
            if (f % 5 == 0) points.push_back(IntrinsicTriangulation::halfedgePoint(3 * f + f % 3, 0.3));
        }
        for (uint32_t v = 0; v < tri.vertexCount(); v++) points.push_back(tri.vertexPoint(v));
        const IntrinsicTriangulation &input = tri.input();
        double worst = 0.0, worstPoint = 0.0;
        size_t inserted = 0;
        for (const SurfacePoint &p : points) {
            const IntrinsicTriangulation::ExtrinsicFrame frame = tri.extrinsicFrame(p);
            worstPoint = std::max(worstPoint, (frame.point.position - tri.toExtrinsic(p).position).norm());
            const uint32_t F = frame.point.face;
            Vec3d corner[3];
            for (int c = 0; c < 3; c++) corner[c] = tri.inputVertices()[input.tail(3 * F + c)].cast<double>();
            const Vec3d n = (corner[1] - corner[0]).cross(corner[2] - corner[0]).normalized();
            // Halfway from X toward a point inside F: the segment stays in F.
            const Vec3d qb = 0.5 * frame.point.bary + 0.5 * RandomBary(rng);
            const Vec3d Q = qb[0] * corner[0] + qb[1] * corner[1] + qb[2] * corner[2];
            const Vec3d X = frame.point.position;
            const Vec2d u = tri.extrinsicToLayout(p, (Q - X) + 0.3 * (Q - X).norm() * n);
            CHECK((u - tri.extrinsicToLayout(frame, Q - X)).norm() <= 1e-12 * (Q - X).norm());
            const SurfacePoint end = tri.trace(p, u).end;
            worst = std::max(worst, (tri.toExtrinsic(end).position - Q).norm());
            if (p.face < tri.faceCount() && tri.faceLevel(p.face) > 0) inserted++;
        }
        CHECK(inserted > 100);
        CHECK(worstPoint < 1e-12);
        CHECK(worst < 1e-9);
        if (worst >= 1e-9) std::printf("extrinsic_to_layout (skewed %d): worst %g\n", skewed ? 1 : 0, worst);
    }
}

namespace
{
// Distance on the unit sphere from x to the arc of the circle at polar angle `rho` around +z, azimuth 0
// to `sweep` (a band swept along a latitude, as the springls sweep it).
double ArcDistance(const Vec3d &x, double rho, double sweep)
{
    const Vec3d u = x.normalized();
    const double polar = std::acos(std::clamp(u.z(), -1.0, 1.0));
    double azimuth = std::atan2(u.y(), u.x());
    if (azimuth < 0.0) azimuth += kTwoPi;
    if (azimuth <= sweep) return std::abs(polar - rho);
    double best = 1e300;
    for (double a : {0.0, sweep}) {
        const Vec3d end(std::sin(rho) * std::cos(a), std::sin(rho) * std::sin(a), std::cos(rho));
        best = std::min(best, std::acos(std::clamp(u.dot(end), -1.0, 1.0)));
    }
    return best;
}

// The refinement criterion of IntrinsicSpringls: faces below maxLevel whose longest edge exceeds 1.01 h_t
// and whose distance lower bound (nearest corner - 0.6 longest) is within the band, refined until none
// qualify (at most maxLevel + 1 passes). Points are relocated.
void RefineBand(IntrinsicTriangulation &tri, int maxLevel, double h, double band, double rho, double sweep, std::vector<SurfacePoint *> &points)
{
    for (int pass = 0; pass <= maxLevel; pass++) {
        std::vector<uint8_t> marks(tri.faceCount(), 0);
        bool any = false;
        for (uint32_t f = 0; f < tri.faceCount(); f++) {
            if (tri.faceLevel(f) >= maxLevel) continue;
            double longest = 0.0, nearest = 1e300;
            for (int c = 0; c < 3; c++) {
                longest = std::max(longest, tri.halfedgeLength(3 * f + c));
                nearest = std::min(nearest, ArcDistance(tri.vertexPosition(tri.tail(3 * f + c)), rho, sweep));
            }
            if (longest > 1.01 * h && nearest - 0.6 * longest <= band) marks[f] = 1;
            any = any || marks[f];
        }
        if (!any) return;
        tri.refine(marks, points);
    }
}
}  // namespace

TEST(refine_incremental_matches_one_pass)
{
    // A band swept along a latitude in k passes (as the springls sweep it, refining each sub-step) and the
    // final band refined at once: both regular (no face above maxLevel, the band's edges near h_t).
    for (const int maxLevel : {2, 3}) {
        const int passes = 12;
        const double rho = 1.0, finalSweep = 2.0;
        IntrinsicTriangulation base = Built(springls_test::Icosphere(3));
        base.flipToDelaunay();
        double mean = 0.0, longestInput = 0.0;
        for (uint32_t e = 0; e < base.edgeCount(); e++) {
            mean += base.length(e);
            longestInput = std::max(longestInput, base.length(e));
        }
        mean /= static_cast<double>(base.edgeCount());
        const double h = mean / std::ldexp(1.0, maxLevel), band = 3.5 * h;
        const auto check = [&](const IntrinsicTriangulation &tri, const char *name) {
            CHECK(Manifold(tri));
            int deepest = 0;
            for (uint32_t f = 0; f < tri.faceCount(); f++) deepest = std::max(deepest, tri.faceLevel(f));
            double shortest = 1e300, longest = 0.0;
            size_t inBand = 0, below09 = 0;
            for (uint32_t e = 0; e < tri.edgeCount(); e++) {
                const uint32_t he = tri.edgeHalfedge(e);
                if (ArcDistance(tri.vertexPosition(tri.tail(he)), rho, finalSweep) > band) continue;
                if (ArcDistance(tri.vertexPosition(tri.head(he)), rho, finalSweep) > band) continue;
                inBand++;
                shortest = std::min(shortest, tri.length(e));
                longest = std::max(longest, tri.length(e));
                below09 += tri.length(e) < 0.9 * h;
            }
            std::printf("  %s (max level %d): %zu faces, deepest level %d, %zu band edges in [%.3f, %.3f] h_t (%.0f%% below 0.9 h_t)\n", name, maxLevel,
                        tri.faceCount(), deepest, inBand, shortest / h, longest / h,
                        100.0 * static_cast<double>(below09) / static_cast<double>(std::max<size_t>(inBand, 1)));
            CHECK(inBand > 500);
            CHECK(deepest <= maxLevel);
            CHECK(shortest >= 0.5 * h);
            CHECK(longest <= 1.2 * h * longestInput / mean);
        };
        // Points spread over the band, relocated through every pass.
        std::mt19937 rng(7);
        IntrinsicTriangulation incremental = base;
        std::vector<SurfacePoint> points;
        for (uint32_t f = 0; f < base.faceCount(); f++) {
            if (ArcDistance(base.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position, rho, finalSweep) < band + mean) {
                for (int n = 0; n < 4; n++) points.push_back({f, RandomBary(rng)});
            }
        }
        std::vector<Vec3d> before;
        std::vector<SurfacePoint *> ptrs;
        for (SurfacePoint &p : points) {
            before.push_back(base.toExtrinsic(p).position);
            ptrs.push_back(&p);
        }
        for (int k = 1; k <= passes; k++) RefineBand(incremental, maxLevel, h, band, rho, finalSweep * k / passes, ptrs);
        check(incremental, "incremental");
        double worst = 0.0;
        for (size_t n = 0; n < points.size(); n++) worst = std::max(worst, (incremental.toExtrinsic(points[n]).position - before[n]).norm());
        std::printf("  relocation: %zu points, worst %.3g\n", points.size(), worst);
        CHECK(worst < 1e-9);
        IntrinsicTriangulation once = base;
        std::vector<SurfacePoint *> none;
        RefineBand(once, maxLevel, h, band, rho, finalSweep, none);
        check(once, "one pass");
    }
}

namespace
{
// Icosphere(4), Delaunay, refined twice around a point.
IntrinsicTriangulation RefinedNearPole(std::vector<SurfacePoint *> &ptrs)
{
    IntrinsicTriangulation tri = Built(springls_test::Icosphere(4));
    tri.flipToDelaunay();
    const Vec3d pole(0, 0, 1);
    for (int pass = 0; pass < 2; pass++) {
        std::vector<uint8_t> marks(tri.faceCount(), 0);
        for (uint32_t f = 0; f < tri.faceCount(); f++) {
            marks[f] = ((tri.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position - pole).norm() < 0.3) ? 1 : 0;
        }
        tri.refine(marks, ptrs);
    }
    return tri;
}
}  // namespace

TEST(reset_to_input_restores_the_input)
{
    std::vector<SurfacePoint *> none;
    IntrinsicTriangulation tri = RefinedNearPole(none);
    const IntrinsicTriangulation &in = tri.input();
    CHECK(tri.faceCount() > in.faceCount());
    tri.resetToInput();
    CHECK(tri.vertexCount() == in.vertexCount());
    CHECK(tri.faceCount() == in.faceCount());
    CHECK(tri.edgeCount() == in.edgeCount());
    double worst = 0.0;
    bool same = true;
    for (uint32_t he = 0; he < 3 * tri.faceCount(); he++) {
        same = same && tri.tail(he) == in.tail(he) && tri.twin(he) == in.twin(he) && tri.edge(he) == in.edge(he);
        worst = std::max(worst, std::abs(tri.signpost(he) - in.signpost(he)));
    }
    CHECK(same);
    CHECK(worst < 1e-12);
    for (uint32_t e = 0; e < tri.edgeCount(); e++) {
        CHECK(std::abs(tri.length(e) - in.length(e)) < 1e-15);
        CHECK(tri.edgeHalfedge(e) == in.edgeHalfedge(e));
    }
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        CHECK(tri.faceLevel(f) == 0);
        CHECK(!tri.bisected(f));
    }
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        CHECK(std::abs(tri.angleSum(v) - in.angleSum(v)) < 1e-12);
        CHECK(tri.vertexHalfedge(v) == in.vertexHalfedge(v));
    }
    CHECK(Manifold(tri));
}

TEST(reset_to_input_keeps_correspondence)
{
    std::vector<SurfacePoint *> none;
    IntrinsicTriangulation tri = RefinedNearPole(none);
    tri.resetToInput();
    std::mt19937 rng(5);
    std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(tri.input().faceCount() - 1));
    double worst = 0.0, rot = 0.0;
    for (int n = 0; n < 200; n++) {
        SurfacePoint p{pick(rng), RandomBary(rng)};
        worst = std::max(worst, (tri.toExtrinsic(p).position - tri.input().toExtrinsic(p).position).norm());
        rot = std::max(rot, std::abs(tri.extrinsicFrame(p).rotation));
    }
    CHECK(worst < 1e-12);
    CHECK(rot < 1e-12);
    // Refine again after the reset: points keep their surface location.
    std::vector<SurfacePoint> points;
    for (uint32_t f = 0; f < tri.faceCount(); f += 7) points.push_back({f, RandomBary(rng)});
    std::vector<Vec3d> before;
    std::vector<SurfacePoint *> ptrs;
    for (auto &p : points) {
        before.push_back(tri.toExtrinsic(p).position);
        ptrs.push_back(&p);
    }
    std::vector<uint8_t> marks(tri.faceCount(), 0);
    for (uint32_t f = 0; f < tri.faceCount(); f++) marks[f] = (tri.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position.z() > 0.6) ? 1 : 0;
    tri.refine(marks, ptrs);
    double again = 0.0;
    for (size_t n = 0; n < points.size(); n++) again = std::max(again, (tri.toExtrinsic(points[n]).position - before[n]).norm());
    CHECK(again < 1e-9);
    CHECK(Manifold(tri));
}
