// Labeled contours (marching triangles per object label) on intrinsic triangulations.
#include "Check.hpp"
#include "IntrinsicTestRegions.hpp"

#include <intrinsic/IntrinsicContour.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

using namespace springls;
using springls_test::MakeRegions;

namespace
{
// Labeled signed values: -d at a label's own vertices, +d elsewhere.
struct LabeledValues {
    springls_test::Regions regions;
    double signedValue(uint32_t v, int label) const { return (regions.labels[v] == label) ? -regions.distance[v] : regions.distance[v]; }
    const std::vector<int> &labels() const { return regions.labels; }
};

LabeledValues Band(const IntrinsicTriangulation &, const springls_test::Regions &r) { return LabeledValues{r}; }

std::vector<IntrinsicCurve> Extract(const IntrinsicTriangulation &tri, const LabeledValues &band, const std::vector<int> &labels)
{
    return ExtractIntrinsicContour(tri, [&](uint32_t v, int l) { return band.signedValue(v, l); }, labels);
}

double Cross(const Vec2d &a, const Vec2d &b) { return a.x() * b.y() - a.y() * b.x(); }

// Whether open segments ab and cd cross (touching at endpoints does not count).
bool ProperlyCross(const Vec2d &a, const Vec2d &b, const Vec2d &c, const Vec2d &d)
{
    constexpr double eps = 1e-12;
    const double d1 = Cross(b - a, c - a), d2 = Cross(b - a, d - a), d3 = Cross(d - c, a - c), d4 = Cross(d - c, b - c);
    return ((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) && ((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps));
}
}  // namespace

TEST(contour_two_labels_closed_curves)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(3));
    const auto r = MakeRegions(tri, [](const Vec3d &p) { return p.z() < 0.1 ? 1 : 2; }, [](const Vec3d &p) { return p.z() - 0.1; });
    const LabeledValues band = Band(tri, r);
    const auto curves = Extract(tri, band, {1, 2});
    CHECK(curves.size() == 2);
    std::map<int, std::set<uint32_t>> edges;
    for (const IntrinsicCurve &c : curves) {
        CHECK(c.closed);
        CHECK(c.points.size() > 20);
        CHECK(c.faces.size() == c.points.size());
        edges[c.label].insert(c.edges.begin(), c.edges.end());
        for (const SurfacePoint &p : c.points) CHECK(std::abs(tri.toExtrinsic(p).position.normalized().z() - 0.1) < 0.02);
    }
    CHECK(edges[1] == edges[2]);
    // Opposite orientations: the region is on the left of each curve. On a sphere, label 1 (south)
    // runs clockwise seen from +z, label 2 counter-clockwise: compare the signed area of the 3D polygons.
    std::map<int, double> spin;
    for (const IntrinsicCurve &c : curves) {
        double s = 0.0;
        for (size_t k = 0; k < c.points.size(); k++) {
            const Vec3d a = tri.toExtrinsic(c.points[k]).position, b = tri.toExtrinsic(c.points[(k + 1) % c.points.size()]).position;
            s += a.x() * b.y() - a.y() * b.x();
        }
        spin[c.label] = s;
    }
    CHECK(spin[1] < 0.0 && spin[2] > 0.0);
}

TEST(contour_t_junction_closed_and_noncrossing)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(3));
    const auto label = [](const Vec3d &p) { return p.z() < -0.2 ? 1 : (p.x() < 0.0 ? 2 : 3); };
    const auto level = [](const Vec3d &p) { return std::min(std::abs(p.z() + 0.2), (p.z() < -0.2) ? 1.0 : std::abs(p.x())); };
    const LabeledValues band = Band(tri, MakeRegions(tri, label, level));
    const auto curves = Extract(tri, band, {1, 2, 3});
    std::set<int> labels;
    for (const IntrinsicCurve &c : curves) {
        CHECK(c.closed);
        labels.insert(c.label);
    }
    CHECK(labels == std::set<int>({1, 2, 3}));
    // Per face, segments of different labels meet only at shared crossings.
    std::map<uint32_t, std::vector<std::pair<int, std::pair<Vec2d, Vec2d>>>> perFace;
    for (const IntrinsicCurve &c : curves) {
        for (size_t k = 0; k < c.faces.size(); k++) {
            const uint32_t f = c.faces[k];
            const Vec2d a = tri.toLayout(ContourPointInFace(tri, c, k, f)), b = tri.toLayout(ContourPointInFace(tri, c, (k + 1) % c.points.size(), f));
            perFace[f].push_back({c.label, {a, b}});
        }
    }
    size_t junctions = 0;
    for (const auto &[f, segs] : perFace) {
        std::set<int> here;
        for (const auto &s : segs) here.insert(s.first);
        if (here.size() == 3) junctions++;
        for (size_t i = 0; i < segs.size(); i++) {
            for (size_t j = i + 1; j < segs.size(); j++) {
                if (segs[i].first == segs[j].first) continue;
                CHECK(!ProperlyCross(segs[i].second.first, segs[i].second.second, segs[j].second.first, segs[j].second.second));
            }
        }
    }
    CHECK(junctions >= 2);
}

TEST(contour_points_lie_on_edges)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::SkewedIcosphere(3, 1.5));
    const auto r = MakeRegions(tri, [](const Vec3d &p) { return p.y() < 0.2 ? 1 : 2; }, [](const Vec3d &p) { return p.y() - 0.2; });
    const auto curves = Extract(tri, Band(tri, r), {1, 2});
    CHECK(!curves.empty());
    for (const IntrinsicCurve &c : curves) {
        CHECK(c.points.size() == c.edges.size() && c.points.size() == c.t.size());
        for (size_t k = 0; k < c.points.size(); k++) {
            CHECK(c.t[k] >= 0.0 && c.t[k] <= 1.0);
            const SurfacePoint expected = IntrinsicTriangulation::halfedgePoint(tri.edgeHalfedge(c.edges[k]), c.t[k]);
            CHECK(c.points[k].face == expected.face && (c.points[k].bary - expected.bary).norm() < 1e-15);
        }
        // Each segment's face has both of the segment's edges.
        for (size_t k = 0; k < c.faces.size(); k++) {
            const uint32_t f = c.faces[k];
            const std::set<uint32_t> fe = {tri.edge(3 * f), tri.edge(3 * f + 1), tri.edge(3 * f + 2)};
            CHECK(fe.count(c.edges[k]) == 1 && fe.count(c.edges[(k + 1) % c.edges.size()]) == 1);
            for (size_t j : {k, (k + 1) % c.points.size()}) {
                const SurfacePoint p = ContourPointInFace(tri, c, j, f);
                CHECK(p.face == f && p.bary.minCoeff() >= 0.0 && std::abs(p.bary.sum() - 1.0) < 1e-12);
                CHECK((tri.toExtrinsic(p).position - tri.toExtrinsic(c.points[j]).position).norm() < 1e-9);
            }
        }
    }
}

TEST(refine_moves_crossing_toward_particle_along_edge)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(3));
    const auto r = MakeRegions(tri, [](const Vec3d &p) { return p.z() < 0.0 ? 1 : 2; }, [](const Vec3d &p) { return p.z(); });
    auto curves = Extract(tri, Band(tri, r), {1});
    CHECK(curves.size() == 1);
    if (curves.size() != 1) return;
    IntrinsicCurve &c = curves[0];
    const size_t k = 5;
    const uint32_t he = tri.edgeHalfedge(c.edges[k]);
    const double t0 = c.t[k];
    // A label-1 particle on the same edge, a little toward its head.
    const double target = std::min(0.95, t0 + 0.3);
    const SurfacePoint particle = IntrinsicTriangulation::halfedgePoint(he, target);
    const double h = tri.meanEdgeLength();
    RefineIntrinsicContour(tri, curves, {particle}, {1}, 1, 2.0 * h, 0.5);
    CHECK(c.edges[k] == tri.edge(he));
    CHECK(c.t[k] > t0 + 0.05 && c.t[k] < target);
    // A particle of another label does not attract.
    auto again = Extract(tri, Band(tri, r), {1});
    RefineIntrinsicContour(tri, again, {particle}, {2}, 1, 2.0 * h, 0.5);
    CHECK(again[0].t[k] == t0);
}

TEST(region_area_hemisphere)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    const double area = RegionArea(tri, [&](uint32_t v) { return static_cast<double>(tri.inputVertices()[v].z()); });
    CHECK(std::abs(area - 2.0 * kPi) < 0.01 * 2.0 * kPi);
    double total = 0.0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) total += tri.faceArea(f);
    CHECK_NEAR(RegionArea(tri, [](uint32_t) { return -1.0; }), total, 1e-9);
    CHECK(RegionArea(tri, [](uint32_t) { return 1.0; }) == 0.0);
}

TEST(contour_area_uses_refined_crossings)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    const auto r = MakeRegions(tri, [](const Vec3d &p) { return p.z() < 0.0 ? 1 : 2; }, [](const Vec3d &p) { return p.z(); });
    const LabeledValues band = Band(tri, r);
    const auto value = [&](uint32_t v, int l) { return band.signedValue(v, l); };
    auto curves = Extract(tri, band, {1, 2});
    // Unrefined: the same as the band's region area.
    for (int l : {1, 2}) CHECK_NEAR(ContourArea(tri, curves, l, value), RegionArea(tri, [&](uint32_t v) { return value(v, l); }), 1e-12);
    // Slide every label-1 crossing halfway to its edge's label-2 end: region 1 grows, region 2 is unchanged.
    const double before = ContourArea(tri, curves, 1, value);
    for (IntrinsicCurve &c : curves) {
        if (c.label != 1) continue;
        for (size_t k = 0; k < c.t.size(); k++) {
            const uint32_t he = tri.edgeHalfedge(c.edges[k]);
            const double end = (band.labels()[tri.head(he)] == 2) ? 1.0 : 0.0;
            c.t[k] = 0.5 * (c.t[k] + end);
        }
    }
    CHECK(ContourArea(tri, curves, 1, value) > before * 1.003);
    CHECK_NEAR(ContourArea(tri, curves, 2, value), RegionArea(tri, [&](uint32_t v) { return value(v, 2); }), 1e-12);
}

TEST(region_pieces_cap_centroid)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    // A cap of angular radius 0.5 about an off-axis direction.
    const Vec3d axis = Vec3d(0.3, -0.5, 0.8).normalized();
    const auto value = [&](uint32_t v) { return std::acos(std::clamp(tri.inputVertices()[v].cast<double>().normalized().dot(axis), -1.0, 1.0)) - 0.5; };
    const std::vector<RegionPiece> pieces = RegionPieces(tri, value);
    double area = 0.0;
    Vec3d sum = Vec3d::Zero();
    for (const RegionPiece &p : pieces) {
        CHECK(p.area > 0.0 && p.centroid.face == p.face);
        CHECK(p.centroid.bary.minCoeff() >= -1e-9 && std::abs(p.centroid.bary.sum() - 1.0) < 1e-9);
        area += p.area;
        sum += p.area * tri.toExtrinsic(p.centroid).position.cast<double>();
    }
    CHECK_NEAR(area, RegionArea(tri, value), 1e-12);
    // The area-weighted centroid lies on the cap's axis, at the cap's centroid depth (1 + cos 0.5) / 2.
    const Vec3d c = sum / area;
    CHECK((c - axis * axis.dot(c)).norm() < 0.05 * tri.meanEdgeLength());
    CHECK(std::abs(axis.dot(c) - 0.5 * (1.0 + std::cos(0.5))) < 0.01);
}
