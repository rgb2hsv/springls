// The Lagrangian reference: the initial contour's points advected by the flow alone.
#include "Check.hpp"
#include "IntrinsicTestRegions.hpp"

#include <intrinsic/IntrinsicContour.hpp>
#include <intrinsic/IntrinsicReference.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

using namespace springls;
using springls_test::MakeRegions;

namespace
{
std::vector<IntrinsicCurve> Contour(const IntrinsicTriangulation &tri, const springls_test::Regions &r, const std::vector<int> &objects)
{
    return ExtractIntrinsicContour(tri, [&](uint32_t v, int l) { return (r.labels[v] == l) ? -r.distance[v] : r.distance[v]; }, objects);
}

std::vector<IntrinsicCurve> StarContour(const IntrinsicTriangulation &tri)
{
    return Contour(tri, springls_test::StarRegions(tri, Vec3d(1, 0.2, 0.5), 0.5), {1, 2});
}

double MaxMovement(const IntrinsicTriangulation &tri, const IntrinsicReference &a, const IntrinsicReference &b)
{
    double worst = 0.0;
    for (size_t c = 0; c < a.curves().size(); c++) {
        for (size_t k = 0; k < a.curves()[c].size(); k++) {
            worst = std::max(worst, (tri.toExtrinsic(a.curves()[c][k]).position - tri.toExtrinsic(b.curves()[c][k]).position).norm());
        }
    }
    return worst;
}
}  // namespace

TEST(reference_initialize_subdivides_each_segment)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    const auto contour = StarContour(tri);
    CHECK(contour.size() == 2);
    IntrinsicReference ref;
    ref.initialize(tri, contour, 4);
    CHECK(ref.curves().size() == contour.size() && ref.labels().size() == contour.size());
    for (size_t c = 0; c < contour.size(); c++) {
        CHECK(ref.curves()[c].size() == 4 * contour[c].points.size());
        CHECK(ref.labels()[c] == contour[c].label);
        for (const SurfacePoint &p : ref.curves()[c]) CHECK(std::abs(tri.toExtrinsic(p).position.norm() - 1.0) < 0.01);
    }
    CHECK(ref.maxGap(tri) < 0.5 * tri.meanEdgeLength());
}

TEST(reference_zero_velocity_stays_put)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicReference ref;
    ref.initialize(tri, StarContour(tri));
    const IntrinsicReference start = ref;
    ref.advance(tri, [](const SurfacePoint &, double) { return Vec2d(0, 0); }, 0.0, 1.0, 0.1, {});
    CHECK(MaxMovement(tri, ref, start) < 1e-12);
}

TEST(reference_rigid_rotation_returns)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    const double h = tri.meanEdgeLength();
    IntrinsicReference ref;
    ref.initialize(tri, StarContour(tri));
    const IntrinsicReference start = ref;
    const auto velocity = [&](const SurfacePoint &p, double) {
        return springls_test::TangentToLayout(tri, p.face, Vec3d(0, 0, 1).cross(springls_test::CornerPosition(tri, p)));
    };
    ref.advance(tri, velocity, 0.0, 2.0 * std::numbers::pi, 0.3 * h, {});
    CHECK(MaxMovement(tri, ref, start) < 0.05 * h);
}

TEST(reference_breaks_are_respected)
{
    // A flat two-sided grid, constant velocity (2, 0) that reverses at t = 0.5; sub-steps (0.35 long) would
    // straddle the reversal without the break.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::DoubledGrid(40, 1.0));
    const auto r = MakeRegions(
        tri, [](const Vec3d &p) { return (Vec2d(p.x() - 20.0, p.y() - 20.0).norm() < 6.0) ? 1 : 0; },
        [](const Vec3d &p) { return Vec2d(p.x() - 20.0, p.y() - 20.0).norm() - 6.0; });
    IntrinsicReference ref;
    ref.initialize(tri, Contour(tri, r, {1}));
    CHECK(!ref.curves().empty());
    const IntrinsicReference start = ref;
    const auto velocity = [&](const SurfacePoint &p, double t) { return springls_test::TangentToLayout(tri, p.face, Vec3d(2, 0, 0)) * (t < 0.5 ? 1.0 : -1.0); };
    ref.advance(tri, velocity, 0.0, 1.0, 0.7, {0.5});
    CHECK(MaxMovement(tri, ref, start) < 1e-6 * tri.meanEdgeLength());
}

TEST(reference_distance_is_zero_on_itself)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    const double h = tri.meanEdgeLength();
    IntrinsicReference ref;
    ref.initialize(tri, StarContour(tri));
    std::vector<SurfacePoint> own;
    for (const auto &curve : ref.curves()) own.insert(own.end(), curve.begin(), curve.end());
    double worst = 0.0;
    for (double d : ref.distancesAt(tri, own, 20.0 * h)) worst = std::max(worst, d);
    CHECK(worst < 1e-9);
    // Far from the reference the distance is the cap.
    const SurfacePoint far = [&] {
        SurfacePoint q;
        q.face = tri.faceCount() / 2;
        q.bary = Vec3d(1, 1, 1) / 3.0;
        return q;
    }();
    const double d = ref.distancesAt(tri, {far}, 20.0 * h)[0];
    CHECK(d > 0.0 && d <= 20.0 * h);
}

TEST(reference_distance_matches_offset_curve)
{
    // A flat two-sided grid cut by the straight line x = 20.5: points 0.3 h beside it are 0.3 h away.
    const springls_test::TestMesh mesh = springls_test::DoubledGrid(40, 1.0);
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(mesh);
    const double h = tri.meanEdgeLength();
    const auto r = MakeRegions(tri, [](const Vec3d &p) { return p.x() < 20.5 ? 1 : 0; }, [](const Vec3d &p) { return p.x() - 20.5; });
    IntrinsicReference ref;
    ref.initialize(tri, Contour(tri, r, {1}));
    CHECK(!ref.curves().empty());
    std::vector<SurfacePoint> queries;
    for (double y = 10.13; y < 30.0; y += 0.97) {
        const Vec3d target(20.5 + 0.3 * h, y, 0.0);
        for (uint32_t f = 0; f < tri.faceCount(); f++) {
            Vec3d c[3];
            for (int q = 0; q < 3; q++) c[q] = tri.inputVertices()[tri.tail(3 * f + q)].cast<double>();
            Vec3d e1 = c[1] - c[0], e2 = c[2] - c[0], d = target - c[0];
            const double det = e1.x() * e2.y() - e1.y() * e2.x();
            const double b1 = (d.x() * e2.y() - d.y() * e2.x()) / det, b2 = (e1.x() * d.y() - e1.y() * d.x()) / det;
            if (b1 >= 0 && b2 >= 0 && b1 + b2 <= 1) {
                SurfacePoint q;
                q.face = f;
                q.bary = Vec3d(1.0 - b1 - b2, b1, b2);
                queries.push_back(q);
                break;
            }
        }
    }
    CHECK(queries.size() > 15);
    for (double d : ref.distancesAt(tri, queries, 20.0 * h)) CHECK(std::abs(d - 0.3 * h) < 1e-6 * h);
}
