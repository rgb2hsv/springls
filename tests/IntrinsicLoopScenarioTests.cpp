// The closed-loop scenario: an object travelling a geodesic circle around the loop center and back.
#include "Check.hpp"
#include "IntrinsicTestRegions.hpp"

#include <intrinsic/IntrinsicLoopScenario.hpp>
#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <tuple>

using namespace springls;

namespace
{
// End-to-end bounds: the runs after Task 4b (regular refinement, contour crossings on the springls,
// springl-side sign correction, no seed springls on segments shorter than an arm at rest; default star at
// the equator, two revolutions), times 1.5. Shape error: the initial signed distance interpolated on the
// input mesh at each contour point, in magnitude (0 at t = 0). Before Task 4b: level 5 max 0 +0.00655,
// 0.089 / 0.493 h_t, 1 contracted, 2 filled; level 4 max 1 +0.0394, 0.162 / 1.073 h_t, 23 contracted,
// 30 filled, 13817 vertices.
// Measured with the schedule's weight cos(πt/T) and sub-steps of 1 h_t (IntrinsicSpringls::kDefaultMaxStep).
// Level 5, max level 0: area error -0.00006, shape error mean 0.011 h_t, max 0.099 h_t, nothing resampled,
// 3.0 s.
constexpr double kLoopAreaBound = 1.5 * 0.00060;  // the bounds of the former schedule (±1, sub-steps of ⅓ h_t)
constexpr double kLoopMeanShapeBound = 1.5 * 0.062;
constexpr double kLoopMaxShapeBound = 1.5 * 0.176;
// Level 4, max level 1 (h_t that of level 5): area error -0.00253, shape error mean 0.016 h_t, max 0.268
// h_t, nothing resampled, 3102 vertices at the end, 2.9 s (the former schedule: -0.00141, 0.054, 0.158).
constexpr int kRefinedLevel = 1;
// Dual-cell flux / perimeter of the projected rotation on the level-4 icosphere (first run 1.05e-4) x 1.5.
constexpr double kDualCellFluxBound = 1.5 * 1.05e-4;
constexpr double kRefinedAreaBound = 1.5 * 0.00253;
constexpr double kRefinedMeanShapeBound = 1.5 * 0.054;
constexpr double kRefinedMaxShapeBound = 1.5 * 0.268;

double PolygonArea(const std::vector<Vec2d> &poly)
{
    double a = 0.0;
    for (size_t k = 0; k < poly.size(); k++) {
        const Vec2d &p = poly[k], &q = poly[(k + 1) % poly.size()];
        a += p.x() * q.y() - p.y() * q.x();
    }
    return 0.5 * a;
}

bool InsidePolygon(const std::vector<Vec2d> &poly, const Vec2d &x)
{
    bool inside = false;
    for (size_t k = 0, j = poly.size() - 1; k < poly.size(); j = k++) {
        const Vec2d &a = poly[k], &b = poly[j];
        if ((a.y() > x.y()) != (b.y() > x.y()) && x.x() < (b.x() - a.x()) * (x.y() - a.y()) / (b.y() - a.y()) + a.x()) inside = !inside;
    }
    return inside;
}

double DistanceToPolygon(const std::vector<Vec2d> &poly, const Vec2d &x)
{
    double best = 1e30;
    for (size_t k = 0; k < poly.size(); k++) {
        const Vec2d a = poly[k], b = poly[(k + 1) % poly.size()];
        const double t = std::clamp((x - a).dot(b - a) / (b - a).squaredNorm(), 0.0, 1.0);
        best = std::min(best, (x - (a + t * (b - a))).norm());
    }
    return best;
}

LoopConfig StarConfig()
{
    LoopConfig config;
    config.object = LoopObject::Star;
    return config;
}

// The analytic rotation at p (in 3D, on the plane of the input face p lies on), and the 3D image of
// the layout velocity there (a short trace along it: exact while it stays in that input face). False
// when the trace leaves the face.
bool RotationAt(const IntrinsicTriangulation &tri, const LoopScenario &scenario, const SurfacePoint &p, Vec3d &expected, Vec3d &image)
{
    const IntrinsicTriangulation &input = tri.input();
    const IntrinsicTriangulation::ExtrinsicPoint x = tri.toExtrinsic(p);
    Vec3d corner[3];
    for (int c = 0; c < 3; c++) corner[c] = tri.inputVertices()[input.tail(3 * x.face + c)].cast<double>();
    const Vec3d n = (corner[1] - corner[0]).cross(corner[2] - corner[0]).normalized();
    const Vec3d w = scenario.angularSpeed() * Vec3d::UnitZ().cross(x.position - scenario.axisCenter());
    expected = w - w.dot(n) * n;
    const Vec2d u = scenario.velocity(tri, p, 0.0);
    if (u.norm() == 0.0) {
        image = Vec3d::Zero();
        return true;
    }
    const double eps = 1e-4 * tri.meanEdgeLength() / u.norm();
    const IntrinsicTriangulation::ExtrinsicPoint y = tri.toExtrinsic(tri.trace(p, eps * u).end);
    if (y.face != x.face) return false;
    image = (y.position - x.position) / eps;
    return true;
}

// Barycentric coordinates of a layout position in face f.
Vec3d BaryOf(const std::array<Vec2d, 3> &P, const Vec2d &x)
{
    Eigen::Matrix2d M;
    M << (P[1] - P[0]), (P[2] - P[0]);
    const Eigen::Vector2d w = M.inverse() * (x - P[0]);
    Vec3d b(1.0 - w[0] - w[1], w[0], w[1]);
    for (int c = 0; c < 3; c++) b[c] = (std::abs(b[c]) < 1e-12) ? 0.0 : b[c];  // corners and edges exactly
    return b / b.sum();
}
}  // namespace

TEST(rotation_velocity_matches_analytic)
{
    IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    LoopScenario scenario;
    std::string error;
    LoopConfig config = StarConfig();
    config.speed = 1.7;
    CHECK(scenario.setup(tri, config, error));
    // The object's center moves at the configured speed.
    CHECK(std::abs(scenario.velocity(scenario.objectCenter(), 0.0).norm() - config.speed) < 1e-3 * config.speed);
    const auto check = [&](const IntrinsicTriangulation &mesh, const char *name) {
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> u(0.05, 1.0);
        double worst = 0.0;
        size_t checked = 0;
        for (uint32_t f = 0; f < mesh.faceCount(); f++) {
            Vec3d b(u(rng), u(rng), u(rng));
            Vec3d expected, image;
            if (!RotationAt(mesh, scenario, SurfacePoint{f, b / b.sum()}, expected, image)) continue;
            worst = std::max(worst, (image - expected).norm());
            checked++;
        }
        std::printf("  %s: %zu points, worst %.3g\n", name, checked, worst);
        CHECK(checked > mesh.faceCount() / 2);
        CHECK(worst < 1e-9 * config.speed);
    };
    check(tri, "input");
    // The same field on a refined copy (points in split, bisected and flipped faces).
    IntrinsicTriangulation refined = tri;
    for (int pass = 0; pass < 2; pass++) {
        std::vector<uint8_t> marks(refined.faceCount(), 0);
        for (uint32_t f = 0; f < refined.faceCount(); f++) {
            const Vec3d x = refined.toExtrinsic(f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)).position;
            marks[f] = (std::abs(x.z()) < 0.4 && x.x() > 0.0) ? 1 : 0;
        }
        refined.refine(marks, {});
    }
    CHECK(refined.vertexCount() > tri.vertexCount() + 1000);
    check(refined, "refined");
}

TEST(rotation_velocity_zero_on_axis)
{
    // The level-4 icosphere has vertexes at both poles, exactly on the rotation axis.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    LoopScenario scenario;
    std::string error;
    CHECK(scenario.setup(tri, StarConfig(), error));
    CHECK(tri.vertexPosition(scenario.center()).isApprox(Vec3d(0, 0, 1), 1e-7));
    size_t poles = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const Vec3d x = tri.vertexPosition(v);
        if (std::abs(std::abs(x.z()) - 1.0) > 1e-7) continue;
        poles++;
        CHECK(scenario.velocity(tri.vertexPoint(v), 0.0).norm() < 1e-6);
    }
    CHECK(poles == 2);
    // Near the axis the speed grows with the distance from it.
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const SurfacePoint p{f, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
        const Vec3d x = tri.toExtrinsic(p).position - scenario.axisCenter();
        const double r = std::hypot(x.x(), x.y());
        if (r > 0.1) continue;
        CHECK(scenario.velocity(p, 0.0).norm() <= scenario.angularSpeed() * r * (1.0 + 1e-9));
        CHECK(scenario.velocity(p, 0.0).norm() >= 0.9 * scenario.angularSpeed() * r);
    }
}

namespace
{
struct Divergence {
    double face = 0.0;  // worst |flux through a face's boundary| / its perimeter
    double cell = 0.0;  // worst |flux through a barycentric dual cell| / its perimeter
};

Divergence RotationDivergence(int level)
{
    IntrinsicTriangulation tri;
    std::string buildError;
    const springls_test::TestMesh mesh = springls_test::Icosphere(level);
    CHECK(tri.build(mesh.vertices, mesh.triangles, buildError));
    LoopScenario scenario;
    std::string error;
    CHECK(scenario.setup(tri, StarConfig(), error));
    // Outward flux of the segment a→b (normal (dy, -dx)) of a counter-clockwise polygon in face f: Simpson's
    // rule (the field is linear in each face of the input mesh).
    const auto out = [&](uint32_t f, const std::array<Vec2d, 3> &P, const Vec2d &a, const Vec2d &b) {
        const int n = 4;
        double sum = 0.0;
        for (int k = 0; k <= n; k++) {
            const double weight = (k == 0 || k == n) ? 1.0 : (k % 2 ? 4.0 : 2.0);
            const Vec2d x = a + (b - a) * (static_cast<double>(k) / n);
            sum += weight * scenario.velocity(SurfacePoint{f, BaryOf(P, x)}, 0.0).dot(Vec2d((b - a).y(), -(b - a).x()));
        }
        return sum / (3.0 * n);
    };
    Divergence d;
    std::vector<double> flux(tri.vertexCount(), 0.0), perimeter(tri.vertexCount(), 0.0);
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const auto P = tri.layout(f);
        const Vec2d centroid = (P[0] + P[1] + P[2]) / 3.0;
        double faceFlux = 0.0, facePerimeter = 0.0;
        for (int c = 0; c < 3; c++) {
            const Vec2d mNext = 0.5 * (P[c] + P[(c + 1) % 3]), mPrev = 0.5 * (P[c] + P[(c + 2) % 3]);
            const uint32_t v = tri.tail(3 * f + c);
            flux[v] += out(f, P, mNext, centroid) + out(f, P, centroid, mPrev);
            perimeter[v] += (centroid - mNext).norm() + (mPrev - centroid).norm();
            faceFlux += out(f, P, P[c], P[(c + 1) % 3]);
            facePerimeter += (P[(c + 1) % 3] - P[c]).norm();
        }
        d.face = std::max(d.face, std::abs(faceFlux) / facePerimeter);
    }
    for (uint32_t v = 0; v < tri.vertexCount(); v++) d.cell = std::max(d.cell, std::abs(flux[v]) / perimeter[v]);
    std::printf("  level %d: worst face flux / perimeter %.3g, dual cell %.3g (h %.4f)\n", level, d.face, d.cell, tri.meanEdgeLength());
    return d;
}
}  // namespace

TEST(rotation_velocity_divergence_free_on_sphere)
{
    // In each face the projected rotation is exactly divergence-free (the in-plane part of a rotation).
    // Across an edge, the two faces' projections of the same 3D vector differ in their normal flux by
    // O(dihedral angle x distance from the edge midpoint): the dual-cell flux is O(h^2) and not zero
    // (first runs: 1.05e-4 at level 4, 2.62e-5 at level 5); bounded by the level-4 run x 1.5,
    // and second order.
    const double speed = StarConfig().speed;
    const Divergence d4 = RotationDivergence(4), d5 = RotationDivergence(5);
    CHECK(d4.face <= 1e-12 * speed && d5.face <= 1e-12 * speed);
    CHECK(d4.cell <= kDualCellFluxBound * speed);
    CHECK(d5.cell <= 0.35 * d4.cell);
}

TEST(loop_period_is_one_revolution_each_way)
{
    constexpr double kPiD = 3.141592653589793;
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    LoopScenario scenario;
    std::string error;
    LoopConfig config = StarConfig();
    config.speed = 2.0;
    CHECK(scenario.setup(tri, config, error));
    // omega = speed / (the object's center's distance from the axis); T = π · 2π / omega (π × one revolution's time).
    const Vec3d x = tri.toExtrinsic(scenario.objectCenter()).position - scenario.axisCenter();
    CHECK_NEAR(scenario.angularSpeed(), config.speed / std::hypot(x.x(), x.y()), 1e-12);
    CHECK_NEAR(scenario.period(), 2.0 * kPiD * kPiD / scenario.angularSpeed(), 1e-12 * scenario.period());
    // On the unit sphere about a pole: π × the time of one lap of the circle at geodesic distance ρ (length 2π sin ρ).
    const double expected = kPiD * (2.0 * kPiD * std::sin(scenario.loopDistance())) / config.speed;
    CHECK(std::abs(scenario.period() - expected) < 0.01 * expected);
    // The schedule's weight cos(πt/T): full speed at the start and end, at rest at T/2, back in the second half.
    const auto schedule = scenario.schedule();
    const double T = scenario.period();
    CHECK_NEAR(schedule.weight(0.0), 1.0, 1e-6);
    CHECK_NEAR(schedule.weight(0.25 * T), std::cos(0.25 * kPiD), 1e-6);
    CHECK(std::abs(schedule.weight(0.5 * T)) < 1e-6);
    CHECK(schedule.weight(0.6 * T) < 0.0f);
    CHECK_NEAR(schedule.weight(T), -1.0, 1e-6);
    // The forward half covers ω ∫₀^{T/2} cos(πt/T) dt = ω T / π = 2π: one revolution.
    CHECK_NEAR(scenario.angularSpeed() * T / kPiD, 2.0 * kPiD, 1e-9);
    CHECK(schedule.endTime == scenario.period());
}

TEST(object_from_outline_single_region)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(5));
    LoopScenario scenario;
    std::string error;
    CHECK(scenario.setup(tri, StarConfig(), error));
    const std::vector<double> &phi = scenario.initialPhi();
    CHECK(phi.size() == tri.vertexCount());
    size_t inside = 0;
    for (double v : phi) inside += v < 0.0;
    CHECK(inside > 100 && inside < tri.vertexCount() / 4);
    // The object's center is inside; the antipode of the loop center outside.
    const IntrinsicTriangulation::ExtrinsicPoint c = tri.toExtrinsic(scenario.objectCenter());
    double atCenter = 0.0;
    for (int q = 0; q < 3; q++) atCenter += c.bary[q] * phi[tri.tail(3 * scenario.objectCenter().face + q)];
    CHECK(atCenter < 0.0);
    uint32_t bottom = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        if (tri.vertexPosition(v).z() < tri.vertexPosition(bottom).z()) bottom = v;
    }
    CHECK(phi[bottom] > 0.0);
    const double area = RegionArea(tri, [&](uint32_t v) { return phi[v]; });
    // The mapped outline's area on the unit sphere: the star is star-shaped about its center, so the
    // area is the integral of (1 - cos R(θ)) dθ (geodesic polar coordinates).
    double outline = 0.0;
    const int n = 4000;
    for (int k = 0; k < n; k++) {
        const double theta = 2.0 * kPi * (k + 0.5) / n;
        outline += (1.0 - std::cos(scenario.radius() * (1.0 + 0.35 * std::cos(5.0 * theta)))) * (2.0 * kPi / n);
    }
    std::printf("  area %.5f, outline %.5f\n", area, outline);
    CHECK(std::abs(area - outline) < 0.02 * outline);
    // Default geometry on the unit sphere: ρ half the antipodal distance, r = ρ / 6.
    CHECK(std::abs(scenario.loopDistance() - 0.5 * kPi) < 2.0 * tri.meanEdgeLength());
    CHECK_NEAR(scenario.radius(), scenario.loopDistance() / 6.0, 1e-12);
}

TEST(loop_fraction_scales_automatic_distance)
{
    // With loopDistance 0, rho = loopFraction x half the distance to the vertex nearest -z; r = rho / 6.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(5));
    LoopConfig config = StarConfig();
    LoopScenario full, quarter;
    std::string error;
    CHECK(full.setup(tri, config, error));
    config.loopFraction = 0.25;
    CHECK(quarter.setup(tri, config, error));
    CHECK_NEAR(quarter.loopDistance(), 0.25 * full.loopDistance(), 1e-12);
    CHECK_NEAR(quarter.radius(), quarter.loopDistance() / 6.0, 1e-12);
    CHECK(quarter.period() < 0.5 * full.period());
    // An explicit loop distance ignores the fraction.
    config.loopDistance = 0.9;
    CHECK(quarter.setup(tri, config, error));
    CHECK_NEAR(quarter.loopDistance(), 0.9, 1e-12);
}

TEST(object_from_outline_flood_fill_respects_outline)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(5));
    const double h = tri.meanEdgeLength();
    const SurfacePoint center{4321, Vec3d(0.3, 0.3, 0.4)};
    const Vec2d xAxis = Vec2d(0.6, 0.8);
    std::vector<Vec2d> square;
    const double s = 0.3;
    for (int k = 0; k < 40; k++) square.emplace_back(-s + 2 * s * k / 40.0, -s);
    for (int k = 0; k < 40; k++) square.emplace_back(s, -s + 2 * s * k / 40.0);
    for (int k = 0; k < 40; k++) square.emplace_back(s - 2 * s * k / 40.0, s);
    for (int k = 0; k < 40; k++) square.emplace_back(-s, s - 2 * s * k / 40.0);
    std::vector<double> phi;
    std::string error;
    CHECK(ObjectFromOutline(tri, center, xAxis, square, phi, error));
    // Local coordinates of each vertex in the object's frame: the sphere's azimuthal equidistant map
    // about the center (its exponential map), with the frame carried from the face's layout to 3D.
    const Vec3d c3 = springls_test::CornerPosition(tri, center).normalized();
    const auto L = tri.layout(center.face);
    const auto P = [&](int k) { return tri.inputVertices()[tri.tail(3 * center.face + k)].cast<double>().eval(); };
    Eigen::Matrix2d Lm;
    Lm << (L[1] - L[0]), (L[2] - L[0]);
    const Eigen::Vector2d w = Lm.inverse() * xAxis.normalized();
    Vec3d X3 = w[0] * (P(1) - P(0)) + w[1] * (P(2) - P(0));
    X3 = (X3 - X3.dot(c3) * c3).normalized();
    const Vec3d Y3 = c3.cross(X3);
    size_t checked = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const Vec3d q = tri.inputVertices()[v].cast<double>().normalized();
        const double angle = std::acos(std::clamp(q.dot(c3), -1.0, 1.0));
        const Vec3d t = q - q.dot(c3) * c3;
        if (angle > 1.0 || t.norm() < 1e-12) continue;
        const Vec2d uv = angle * Vec2d(t.dot(X3), t.dot(Y3)) / t.norm();
        if (DistanceToPolygon(square, uv) < 1.5 * h) continue;  // too close to call
        checked++;
        // One region, no split line: negative inside, the distance to the outline.
        CHECK((phi[v] < 0.0) == InsidePolygon(square, uv));
        const double expected = DistanceToPolygon(square, uv);
        // Exact next to the outline; first-order fast-marching error grows with distance.
        CHECK(std::abs(std::abs(phi[v]) - expected) < 0.2 * h + 0.05 * expected || expected > 4.0 * h);
    }
    CHECK(checked > 100);
}

TEST(puzzle_outline_has_tabs_and_sockets)
{
    const double r = 1.0;
    const std::vector<Vec2d> outline = PuzzlePieceOutline(r, 400);
    // Tabs and sockets are the same disk shape, so the area is the square's.
    CHECK(std::abs(PolygonArea(outline) - 4.0 * r * r) < 0.01 * 4.0 * r * r);
    double maxX = -1e30, maxY = -1e30;
    for (const Vec2d &p : outline) {
        maxX = std::max(maxX, p.x());
        maxY = std::max(maxY, p.y());
    }
    CHECK_NEAR(maxX, 1.6 * r, 0.01 * r);  // front tab
    CHECK_NEAR(maxY, 1.6 * r, 0.01 * r);  // top tab
    // Sockets: the back and bottom sides are cut in to 0.4 r from the center.
    CHECK(InsidePolygon(outline, Vec2d(-0.35 * r, 0.0)) && !InsidePolygon(outline, Vec2d(-0.45 * r, 0.0)));
    CHECK(InsidePolygon(outline, Vec2d(0.0, -0.35 * r)) && !InsidePolygon(outline, Vec2d(0.0, -0.45 * r)));
    CHECK(PolygonArea(outline) > 0.0);  // counter-clockwise
}

namespace
{
// Where a fill or contraction happened on the reference shape: the event's position rotated back to
// t = 0 (the exact flow on the sphere), the nearest point of the initial outline mapped onto the
// surface, and the outline's turning there over +-1 h_t of arc (counter-clockwise outline: positive
// convex, negative concave).
enum class Bend {
    Convex,
    Straight,
    Concave
};
constexpr double kBendAngle = 0.15;  // radians of turning over the 2 h_t window

struct ClassifiedEvent {
    IntrinsicSpringls::ResampleEvent event;
    Bend bend = Bend::Straight;
    double referenceDistance = 0.0;      // h_t, from the rotated-back position to the mapped outline
    Vec2d outlinePoint = Vec2d::Zero();  // the nearest outline point, in units of r
    double turning = 0.0;
};

struct LoopRun {
    LoopScenario::Metrics initial, metrics;  // after init, at the end
    int steps = 0;
    double seconds = 0.0;
    size_t contracted = 0, filled = 0, leaks = 0, vertices = 0;
    size_t initialSpringls = 0;
    size_t firstStepVertices = 0, maxVertices = 0, rebuilds = 0;  // after the first step, the most over the steps
    std::vector<ClassifiedEvent> events;
};

std::vector<ClassifiedEvent> ClassifyEvents(const IntrinsicTriangulation &tri, const LoopScenario &scenario, LoopObject object, double h,
                                            const std::vector<IntrinsicSpringls::ResampleEvent> &events)
{
    // The outline densely (about 20 samples per h_t), mapped by the exponential map as ObjectFromOutline.
    const double r = scenario.radius();
    const int dense = std::max(400, static_cast<int>(20.0 * 2.0 * 6.3 * 1.6 * r / h));
    const std::vector<Vec2d> outline = (object == LoopObject::Star) ? StarOutline(r, dense) : PuzzlePieceOutline(r, dense / 8);
    const size_t n = outline.size();
    const Vec2d ex = scenario.velocity(scenario.objectCenter(), 0.0).normalized(), ey(-ex.y(), ex.x());
    std::vector<Vec3d> mapped(n);
    for (size_t k = 0; k < n; k++) mapped[k] = tri.toExtrinsic(Trace(tri, scenario.objectCenter(), outline[k].x() * ex + outline[k].y() * ey).end).position;
    std::vector<double> arc(n + 1, 0.0);
    for (size_t k = 0; k < n; k++) arc[k + 1] = arc[k] + (outline[(k + 1) % n] - outline[k]).norm();
    const double perimeter = arc[n];
    // Turning per sample: the direction change between the chords to the points 1 h_t of arc behind and ahead.
    const auto at = [&](double s) {
        s = std::fmod(s, perimeter);
        if (s < 0.0) s += perimeter;
        const size_t k = static_cast<size_t>(std::upper_bound(arc.begin(), arc.end(), s) - arc.begin()) - 1;
        const double w = (s - arc[k]) / std::max(arc[k + 1] - arc[k], 1e-300);
        return Vec2d((1.0 - w) * outline[k % n] + w * outline[(k + 1) % n]);
    };
    std::vector<double> turning(n);
    for (size_t k = 0; k < n; k++) {
        const Vec2d back = outline[k] - at(arc[k] - h), ahead = at(arc[k] + h) - outline[k];
        turning[k] = std::atan2(back.x() * ahead.y() - back.y() * ahead.x(), back.dot(ahead));
    }
    const double T = scenario.period(), omega = scenario.angularSpeed();
    const Vec3d c = scenario.axisCenter();
    std::vector<ClassifiedEvent> out;
    for (const IntrinsicSpringls::ResampleEvent &e : events) {
        // The angle turned by time t under the weight cos(πs/T): ω ∫₀^t cos(πs/T) ds.
        const double theta = omega * (T / std::acos(-1.0)) * std::sin(std::acos(-1.0) * e.time / T);
        const Vec3d x = e.position - c;
        const Vec3d x0 = c + Vec3d(std::cos(theta) * x.x() + std::sin(theta) * x.y(), -std::sin(theta) * x.x() + std::cos(theta) * x.y(), x.z());
        size_t best = 0;
        double bestDistance = 1e300;
        for (size_t k = 0; k < n; k++) {
            const double d = (mapped[k] - x0).norm();
            if (d < bestDistance) {
                bestDistance = d;
                best = k;
            }
        }
        ClassifiedEvent ce;
        ce.event = e;
        ce.referenceDistance = bestDistance / h;
        ce.outlinePoint = outline[best] / r;
        ce.turning = turning[best];
        ce.bend = (turning[best] > kBendAngle) ? Bend::Convex : (turning[best] < -kBendAngle) ? Bend::Concave : Bend::Straight;
        out.push_back(ce);
    }
    return out;
}

const char *KindName(IntrinsicSpringls::ResampleEvent::Kind kind)
{
    switch (kind) {
        case IntrinsicSpringls::ResampleEvent::Kind::Fill: return "fill";
        case IntrinsicSpringls::ResampleEvent::Kind::ContractFar: return "contract far";
        case IntrinsicSpringls::ResampleEvent::Kind::ContractShortArm: return "contract short arm";
        case IntrinsicSpringls::ResampleEvent::Kind::ContractLongArm: return "contract long arm";
    }
    return "?";
}

// Per kind and bend: count, distance quartiles (h_t); fills also the hole widths.
void PrintEvents(const std::vector<ClassifiedEvent> &events)
{
    using Kind = IntrinsicSpringls::ResampleEvent::Kind;
    const char *bendNames[3] = {"convex", "straight", "concave"};
    for (Kind kind : {Kind::Fill, Kind::ContractFar, Kind::ContractShortArm, Kind::ContractLongArm}) {
        for (int b = 0; b < 3; b++) {
            std::vector<double> d, sep, ref;
            for (const ClassifiedEvent &e : events) {
                if (e.event.kind != kind || static_cast<int>(e.bend) != b) continue;
                d.push_back(e.event.distance);
                sep.push_back(e.event.separation);
                ref.push_back(e.referenceDistance);
            }
            if (d.empty()) continue;
            const auto q = [](std::vector<double> v, double f) {
                std::sort(v.begin(), v.end());
                return v[std::min(v.size() - 1, static_cast<size_t>(f * static_cast<double>(v.size())))];
            };
            size_t real = 0;
            for (double s : sep) real += s > 0.5;
            std::printf("    %-18s %-8s n %3zu  distance min/med/max %.3f %.3f %.3f  sep med %.3f (>0.5: %zu)  ref med %.3f max %.3f\n", KindName(kind),
                        bendNames[b], d.size(), q(d, 0.0), q(d, 0.5), q(d, 1.0), q(sep, 0.5), real, q(ref, 0.5), q(ref, 1.0));
        }
    }
}

// The whole loop with the given object at the given base level and refinement depth.
LoopRun RunLoop(int level, int maxLevel, LoopObject object = LoopObject::Star, int maxSteps = 100000, bool coarsening = true)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(level));
    LoopScenario scenario;
    std::string error;
    LoopConfig config;
    config.object = object;
    CHECK(scenario.setup(tri, config, error));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(maxLevel);
    sim.setCoarsening(coarsening);
    sim.setInitialRegion(scenario.initialPhi());
    // The simulation's points live on its refined copy of the triangulation.
    sim.setVelocity([&](const SurfacePoint &p, double t) { return scenario.velocity(sim.triangulation(), p, t); }, scenario.schedule());
    CHECK(sim.init(error));
    LoopRun run;
    run.initial = scenario.measure(sim);
    run.initialSpringls = sim.springls().size();
    const auto t0 = std::chrono::steady_clock::now();
    for (bool more = true; more && run.steps < maxSteps;) {
        more = sim.step();
        if (more) run.steps++;
        const size_t vertices = sim.triangulation().vertexCount();
        if (sim.iteration() == 1) run.firstStepVertices = vertices;
        run.maxVertices = std::max(run.maxVertices, vertices);
    }
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    run.metrics = scenario.measure(sim);
    run.contracted = sim.contractedTotal();
    run.filled = sim.filledTotal();
    run.leaks = sim.leakCount();
    run.vertices = sim.triangulation().vertexCount();
    run.rebuilds = sim.rebuildCount();
    run.events = ClassifyEvents(tri, scenario, object, sim.meanEdgeLength(), sim.resampleEvents());
    const LoopScenario::Metrics &m = run.metrics;
    std::printf(
        "  loop (%s, level %d, max level %d): %d steps, %.1f s, t = %.3f of T = %.3f, area error %.5f, shape error mean %.3f max %.3f h_t (at t = 0: "
        "area %.5f, shape %.3f / %.3f h_t), springls %zu -> %zu, contracted %zu, filled %zu, leaks %zu, vertices %zu (after the first step %zu, most %zu, %zu "
        "rebuilds)\n",
        object == LoopObject::Star ? "star" : "puzzle", level, maxLevel, run.steps, run.seconds, m.time, scenario.period(), m.relativeAreaErrors[0],
        m.meanShapeError, m.maxShapeError, run.initial.relativeAreaErrors[0], run.initial.meanShapeError, run.initial.maxShapeError, run.initialSpringls,
        m.springls, run.contracted, run.filled, run.leaks, run.vertices, run.firstStepVertices, run.maxVertices, run.rebuilds);
    PrintEvents(run.events);
    if (std::getenv("SPRINGLS_EVENTS")) {
        for (const ClassifiedEvent &e : run.events)
            std::printf("      %-18s t %7.3f  at (%6.3f, %6.3f) turn %6.3f  distance %.3f sep %.3f ref %.3f\n", KindName(e.event.kind), e.event.time,
                        e.outlinePoint.x(), e.outlinePoint.y(), e.turning, e.event.distance, e.event.separation, e.referenceDistance);
    }
    if (maxSteps >= 100000) CHECK_NEAR(m.time, scenario.period(), 1e-9);
    CHECK(m.areas.size() == 1 && m.relativeAreaErrors.size() == 1);
    return run;
}
// RunLoop's full loops, run once per process (several tests read the same runs).
const LoopRun &CachedLoop(int level, int maxLevel, LoopObject object)
{
    static std::map<std::tuple<int, int, int>, LoopRun> runs;
    const auto key = std::make_tuple(level, maxLevel, static_cast<int>(object));
    auto it = runs.find(key);
    if (it == runs.end()) it = runs.emplace(key, RunLoop(level, maxLevel, object)).first;
    return it->second;
}
}  // namespace

TEST(loop_end_to_end_star_on_sphere)
{
    // The input resolution (max level 0).
    const LoopRun &run = CachedLoop(5, 0, LoopObject::Star);
    CHECK(std::abs(run.metrics.relativeAreaErrors[0]) < kLoopAreaBound);
    CHECK(run.metrics.meanShapeError < kLoopMeanShapeBound);
    CHECK(run.metrics.maxShapeError < kLoopMaxShapeBound);
}

TEST(loop_end_to_end_star_refined)
{
    // Adaptive refinement near the springls (level 4 base, max level kRefinedLevel).
    const LoopRun &run = CachedLoop(4, kRefinedLevel, LoopObject::Star);
    CHECK(std::abs(run.metrics.relativeAreaErrors[0]) < kRefinedAreaBound);
    CHECK(run.metrics.meanShapeError < kRefinedMeanShapeBound);
    CHECK(run.metrics.maxShapeError < kRefinedMaxShapeBound);
    CHECK(run.leaks == 0);
}

TEST(loop_fills_are_rare)
{
    // The star and the puzzle piece (concave sockets and tab necks), uniform and refined: fills and
    // contractions together at most 2% of the springls over the whole loop, and every fill closes a real
    // gap (the springls there more than fillDistance apart), never a contour that drifted off them.
    for (const LoopObject object : {LoopObject::Star, LoopObject::PuzzlePiece}) {
        for (const auto &[level, maxLevel] : {std::make_pair(5, 0), std::make_pair(4, kRefinedLevel)}) {
            const LoopRun &run = CachedLoop(level, maxLevel, object);
            const size_t resampled = run.contracted + run.filled;
            size_t falseFills = 0;
            for (const ClassifiedEvent &e : run.events) {
                if (e.event.kind == IntrinsicSpringls::ResampleEvent::Kind::Fill && !(e.event.separation > 0.5)) falseFills++;
            }
            std::printf("  %s level %d max %d: resampled %zu of %zu springls (%.1f%%), fills without a gap %zu\n",
                        object == LoopObject::Star ? "star" : "puzzle", level, maxLevel, resampled, run.initialSpringls,
                        100.0 * static_cast<double>(resampled) / static_cast<double>(run.initialSpringls), falseFills);
            CHECK(100 * resampled <= 2 * run.initialSpringls);
            CHECK(falseFills == 0);
            CHECK(run.leaks == 0);
        }
    }
}

TEST(object_from_outline_large_object)
{
    // Seeds far from the object's center (more than LocalLog's reach) still find their vertices.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(5));
    LoopScenario scenario;
    std::string error;
    LoopConfig config = StarConfig();
    config.radius = 0.35;
    CHECK(scenario.setup(tri, config, error));
    if (!error.empty()) std::printf("  setup: %s\n", error.c_str());
    size_t inside = 0;
    for (double v : scenario.initialPhi()) inside += v < 0.0;
    CHECK(inside > 250 && inside < tri.vertexCount() / 2);  // about 3% of the sphere
}

TEST(object_from_outline_larger_than_half)
{
    // A circle of geodesic radius 0.65π about a vertex of the unit sphere encloses (1 - cos 0.65π) / 2 ≈ 73%
    // of it: the object is the larger side, and the outline's orientation (counter-clockwise: inside on
    // its left) says which side is inside, not the size of the flooded region.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(5));
    const SurfacePoint center = tri.vertexPoint(0);
    const double radius = 0.65 * kPi;
    for (const bool clockwise : {false, true}) {
        std::vector<Vec2d> circle;
        const int n = 2000;
        for (int k = 0; k < n; k++) {
            const double theta = (clockwise ? -1.0 : 1.0) * 2.0 * kPi * k / n;
            circle.push_back(radius * Vec2d(std::cos(theta), std::sin(theta)));
        }
        std::vector<double> phi;
        std::string error;
        CHECK(ObjectFromOutline(tri, center, Vec2d(1, 0), circle, phi, error));
        if (!error.empty()) std::printf("  %s: %s\n", clockwise ? "clockwise" : "counter-clockwise", error.c_str());
        if (phi.size() != tri.vertexCount()) continue;
        const double area = RegionArea(tri, [&](uint32_t v) { return phi[v]; });
        const double expected = 2.0 * kPi * (1.0 - std::cos(radius));
        std::printf("  %s: area %.4f, expected %.4f (%.1f%% of the sphere)\n", clockwise ? "clockwise" : "counter-clockwise", area, expected,
                    100.0 * expected / (4.0 * kPi));
        CHECK(std::abs(area - expected) < 0.01 * expected);
        CHECK(phi[0] < 0.0);
        uint32_t antipode = 0;
        for (uint32_t v = 0; v < tri.vertexCount(); v++) {
            if (tri.vertexPosition(v).dot(tri.vertexPosition(0)) < tri.vertexPosition(antipode).dot(tri.vertexPosition(0))) antipode = v;
        }
        CHECK(phi[antipode] > 0.0);
    }
}

TEST(loop_resampling_diagnostics)
{
    // Not a check: the fill and contraction triggers of loops, for the record. Runs only with
    // SPRINGLS_DIAG set: "all" (star and puzzle, level 5 max 0 and level 4 max 1) or
    // "object,level,maxLevel[,steps]" (object star or puzzle).
    const char *diag = std::getenv("SPRINGLS_DIAG");
    if (!diag) {
        std::printf("  (skipped: set SPRINGLS_DIAG)\n");
        return;
    }
    if (std::string(diag) == "all") {
        for (const LoopObject object : {LoopObject::Star, LoopObject::PuzzlePiece}) {
            RunLoop(5, 0, object);
            RunLoop(4, 1, object);
        }
        return;
    }
    char name[16] = {0};
    int level = 5, maxLevel = 0, steps = 100000;
    if (std::sscanf(diag, "%15[a-z],%d,%d,%d", name, &level, &maxLevel, &steps) >= 3)
        RunLoop(level, maxLevel, std::string(name) == "star" ? LoopObject::Star : LoopObject::PuzzlePiece, steps);
}

TEST(loop_puzzle_corners_keep_springls_in_line)
{
    // The puzzle piece's sharp corners (90 degree square corners, the ~120 degree concave tab necks) at
    // the start of the refined loop: the springls stay a chain (each arm vertex links to the opposite end of
    // a neighbor, v1 to v0: never end to same end, a springl lying across or against the chain), and
    // nothing is resampled.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    LoopScenario scenario;
    std::string error;
    LoopConfig config;
    config.object = LoopObject::PuzzlePiece;
    CHECK(scenario.setup(tri, config, error));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    sim.setInitialRegion(scenario.initialPhi());
    sim.setVelocity([&](const SurfacePoint &p, double t) { return scenario.velocity(sim.triangulation(), p, t); }, scenario.schedule());
    CHECK(sim.init(error));
    size_t worstStep = 0, crosswise = 0;
    double worstCos = 1.0;
    for (int step = 0; step < 30; step++) {
        sim.step();
        const IntrinsicTriangulation &t = sim.triangulation();
        const auto &springls = sim.springls();
        const auto &nearest = sim.nearestNeighbors();
        if (nearest.size() != 2 * springls.size()) continue;
        std::vector<Vec3d> chord(springls.size());
        for (size_t s = 0; s < springls.size(); s++) chord[s] = (t.toExtrinsic(springls[s].v1).position - t.toExtrinsic(springls[s].v0).position).normalized();
        for (size_t i = 0; i < nearest.size(); i++) {
            if (nearest[i] < 0) continue;
            const double c = chord[i / 2].dot(chord[static_cast<size_t>(nearest[i]) / 2]);
            if (c < worstCos) {
                worstCos = c;
                worstStep = static_cast<size_t>(step);
            }
            crosswise += (i % 2) == (static_cast<size_t>(nearest[i]) % 2);
        }
    }
    std::printf("  30 steps: worst neighbor chord cosine %.3f (step %zu), %zu same-end links, contracted %zu, filled %zu\n", worstCos, worstStep, crosswise,
                sim.contractedTotal(), sim.filledTotal());
    CHECK(crosswise == 0);
    CHECK(sim.contractedTotal() + sim.filledTotal() == 0);
}

TEST(loop_chart_cache_follows_the_springls)
{
    // A quarter of the loop (the star sweeps several of its own sizes): the chart caches stay as large as the
    // band around the springls instead of growing with the swept area.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(5));
    LoopScenario scenario;
    std::string error;
    CHECK(scenario.setup(tri, StarConfig(), error));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(0);
    sim.setInitialRegion(scenario.initialPhi());
    sim.setVelocity([&](const SurfacePoint &p, double t) { return scenario.velocity(sim.triangulation(), p, t); }, scenario.schedule());
    CHECK(sim.init(error));
    size_t early = 0, peak = 0;
    for (int step = 0; step < 80; step++) {
        sim.step();
        if (step == 10) early = sim.chartCacheSize();
        if (step >= 10) peak = std::max(peak, sim.chartCacheSize());
    }
    std::printf("  chart cache: %zu charts after 10 steps, peak %zu over 80 steps (%zu faces)\n", early, peak, sim.triangulation().faceCount());
    CHECK(early > 0);
    CHECK(peak <= early + early / 2);
}

TEST(springls_fill_skips_short_segments)
{
    // Gaps opened in a star on a refined mesh, closed by fill: no fill springl on a contour segment shorter
    // than an arm at rest (it would have no reliable direction, as a short seed).
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    LoopScenario scenario;
    std::string error;
    CHECK(scenario.setup(tri, StarConfig(), error));
    size_t fills = 0, shortFills = 0;
    double shortest = 1e300;
    for (int offset = 0; offset < 100; offset += 1) {
        IntrinsicSpringls sim(tri);
        sim.setMaxLevel(1);
        sim.setInitialRegion(scenario.initialPhi());
        CHECK(sim.init(error));
        std::vector<uint8_t> keep(sim.springls().size(), 1);
        for (int k = 0; k < 4; k++) keep[(static_cast<size_t>(offset) + k) % keep.size()] = 0;
        sim.removeSpringls(keep);
        sim.step();
        for (const IntrinsicSpringls::ResampleEvent &e : sim.resampleEvents()) {
            if (e.kind != IntrinsicSpringls::ResampleEvent::Kind::Fill) continue;
            fills++;
            shortest = std::min(shortest, e.length);
            shortFills += e.length < sim.tuning.armRestLength;
        }
    }
    std::printf("  %zu fills, %zu on segments shorter than the arm rest length, shortest %.3f h_t\n", fills, shortFills, shortest);
    CHECK(fills > 10);
    CHECK(shortFills == 0);
}

TEST(loop_vertex_count_follows_band)
{
    // The refined loop (level 4, max level 2) with and without coarsening: with it, the refined region
    // follows the band (the vertex count stays near that after the first step); without, it grows with the
    // swept area.
    const LoopRun on = RunLoop(4, 2, LoopObject::Star, 100000, true);
    const LoopRun off = RunLoop(4, 2, LoopObject::Star, 100000, false);
    std::printf(
        "  vertex count: coarsening on: first step %zu, most %zu (%.2fx), %zu rebuilds, %.1f s, area error %.5f; off: first step %zu, most %zu "
        "(%.2fx), %.1f s, area error %.5f\n",
        on.firstStepVertices, on.maxVertices, static_cast<double>(on.maxVertices) / static_cast<double>(on.firstStepVertices), on.rebuilds, on.seconds,
        on.metrics.relativeAreaErrors[0], off.firstStepVertices, off.maxVertices,
        static_cast<double>(off.maxVertices) / static_cast<double>(off.firstStepVertices), off.seconds, off.metrics.relativeAreaErrors[0]);
    CHECK(on.rebuilds >= 1);
    CHECK(2 * on.maxVertices <= 3 * on.firstStepVertices);
    CHECK(2 * off.maxVertices > 3 * off.firstStepVertices);
    CHECK(off.rebuilds == 0);
    CHECK(std::abs(on.metrics.relativeAreaErrors[0]) < 0.01);
}
