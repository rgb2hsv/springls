// Level set reconstructed from springls on the mesh.
#include "Check.hpp"
#include "IntrinsicTestRegions.hpp"

#include <intrinsic/IntrinsicSignField.hpp>
#include <intrinsic/IntrinsicTrace.hpp>

#include <cmath>
#include <numbers>

using namespace springls;

namespace
{
using Polyline = std::vector<Vec2d>;

// Segments of a closed polygon given in the layout of `face` around its center, subdivided to
// pieces of at most `piece`, mapped onto the surface by the exponential map.
SpringlSegments ClosedOutline(const IntrinsicTriangulation &tri, uint32_t face, const Polyline &polygon, double piece, double gapLength = 0.0)
{
    const SurfacePoint center{face, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    std::vector<SurfacePoint> points;
    std::vector<double> arc;  // polygon arc length at each point
    double length = 0.0;
    for (size_t i = 0; i < polygon.size(); i++) {
        const Vec2d a = polygon[i], b = polygon[(i + 1) % polygon.size()];
        const int n = std::max(1, static_cast<int>(std::ceil((b - a).norm() / piece)));
        for (int k = 0; k < n; k++) {
            points.push_back(Trace(tri, center, a + (b - a) * (double(k) / n)).end);
            arc.push_back(length + (b - a).norm() * k / n);
        }
        length += (b - a).norm();
    }
    SpringlSegments s;
    for (size_t i = 0; i < points.size(); i++) {
        if (arc[i] < gapLength) continue;  // the first gapLength of the outline is left open
        s.segments.emplace_back(points[i], points[(i + 1) % points.size()]);
    }
    return s;
}

// Amplitude 0.2: where the outline's curvature radius drops below a mesh edge, the linear zero
// crossing of a distance field is off by more than a tenth of an edge, whatever the sign logic.
Polyline Star(double r, int samples = 200, double amplitude = 0.2)
{
    Polyline p;
    for (int i = 0; i < samples; i++) {
        const double t = 2.0 * std::numbers::pi * i / samples, R = r * (1.0 + amplitude * std::cos(5.0 * t));
        p.push_back(R * Vec2d(std::cos(t), std::sin(t)));
    }
    return p;
}
bool InsideStar(const Vec2d &x, double r, double amplitude = 0.2)
{
    const double t = std::atan2(x.y(), x.x());
    return x.norm() < r * (1.0 + amplitude * std::cos(5.0 * t));
}

// Distance from the vertex to the segments, measured in a chart centered on the vertex's face.
double ExactDistance(const IntrinsicTriangulation &tri, const SpringlSegments &s, const SurfacePoint &p, int depth, double limit)
{
    const LocalChart c(tri, p.face, depth);
    const Vec2d x = c.position(p);
    double best = std::numeric_limits<double>::infinity();
    for (const auto &[a, b] : s.segments) {
        if (!c.contains(a.face) || !c.contains(b.face)) continue;
        const Vec2d pa = c.position(a), pb = c.position(b), ab = pb - pa;
        const double t = std::clamp((x - pa).dot(ab) / std::max(1e-300, ab.squaredNorm()), 0.0, 1.0);
        best = std::min(best, (x - (pa + t * ab)).norm());
    }
    return std::min(best, limit);
}

// Every mesh edge whose endpoints differ in sign: the linear zero crossing's distance to the segments.
double WorstCrossing(const IntrinsicTriangulation &tri, const SpringlSegments &s, const std::vector<double> &phi, int *count = nullptr)
{
    double worst = 0.0;
    int n = 0;
    for (uint32_t he = 0; he < 3 * tri.faceCount(); he++) {
        const uint32_t a = tri.tail(he), b = tri.head(he);
        if (a > b || (phi[a] < 0) == (phi[b] < 0)) continue;
        const double t = phi[a] / (phi[a] - phi[b]);
        const double d = ExactDistance(tri, s, IntrinsicTriangulation::halfedgePoint(he, t), 5, 1e9);
        worst = std::max(worst, d);
        n++;
    }
    if (count) *count = n;
    return worst;
}

// The side of the nearest segment (the region on its left), with the summed normals of the segments
// tied at a shared end.
SpringlSide NearestSide(const IntrinsicTriangulation &tri, const SpringlSegments &s)
{
    const double h = tri.meanEdgeLength();
    return [&tri, &s, h](uint32_t v, double reach) {
        const SurfacePoint p = tri.vertexPoint(v);
        const LocalChart c(tri, p.face, 8);
        const Vec2d x = c.position(p);
        double best = reach, sum = 0.0;
        for (const auto &[a, b] : s.segments) {
            if (!c.contains(a.face) || !c.contains(b.face)) continue;
            const Vec2d pa = c.position(a), pb = c.position(b), ab = pb - pa;
            const double t = std::clamp((x - pa).dot(ab) / ab.squaredNorm(), 0.0, 1.0);
            const Vec2d foot = pa + t * ab;
            const double d = (x - foot).norm();
            const double dot = Vec2d(-ab.y(), ab.x()).normalized().dot(x - foot);
            if (d < best - 1e-12 * h) {
                best = d;
                sum = dot;
            } else if (d <= best + 1e-12 * h) {
                sum += dot;
            }
        }
        return (sum > 0.0) ? -1 : (sum < 0.0) ? 1 : 0;
    };
}

IntrinsicTriangulation Mesh(int level) { return springls_test::DelaunaySurface(springls_test::Icosphere(level)); }
}  // namespace

TEST(sign_field_star_inside_outside)
{
    const IntrinsicTriangulation tri = Mesh(5);
    const double h = tri.meanEdgeLength(), r = 0.6;
    const SpringlSegments s = ClosedOutline(tri, 0, Star(r), 0.5 * h);
    const SignField f = ReconstructSignField(tri, s, {1.5 * h, 4.0 * h});
    CHECK(f.hasInside);
    // Vertices well away from the outline (beyond a capsule) have the sign of the star. The star's
    // layout frame is face 0's; classify through the exponential-map inverse (LocalLog) from the center.
    const SurfacePoint center{0, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    int checked = 0, wrong = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const double d = ExactDistance(tri, s, tri.vertexPoint(v), 6, 1e9);
        if (d < 1.5 * h) continue;
        const std::optional<Vec2d> x = LocalLog(tri, center, tri.vertexPoint(v), 12);
        if (!x) continue;
        checked++;
        if (InsideStar(*x, r) != (f.phi[v] < 0)) wrong++;
    }
    CHECK(checked > 200);
    CHECK(wrong == 0);
    int crossings = 0;
    CHECK(WorstCrossing(tri, s, f.phi, &crossings) <= 0.1 * h);
    CHECK(crossings > 50);
}

TEST(sign_field_keeps_thin_filament)
{
    // A closed strip 1.5 h wide: thinner than two capsules and than the mesh resolves (about one
    // vertex row), so the flood never reaches inside and the filament lives on the local maxima of U.
    const IntrinsicTriangulation tri = Mesh(4);
    const double h = tri.meanEdgeLength();
    const double w = 0.75 * h, L = 6.0 * h;
    const Polyline strip = {{-L, -w}, {L, -w}, {L, w}, {-L, w}};
    const SpringlSegments s = ClosedOutline(tri, 0, strip, 0.5 * h);
    const SignField f = ReconstructSignField(tri, s, {1.5 * h, 4.0 * h});
    CHECK(f.hasInside);
    const SurfacePoint center{0, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    int inside = 0, strayInside = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        if (f.phi[v] >= 0.0) continue;
        inside++;
        // Inside vertices lie in the strip (a margin of half an edge for the position error).
        const std::optional<Vec2d> x = LocalLog(tri, center, tri.vertexPoint(v), 12);
        if (!x || std::abs(x->x()) > L + 0.5 * h || std::abs(x->y()) > w + 0.5 * h) strayInside++;
    }
    CHECK(inside >= 4);
    CHECK(strayInside == 0);
    // Most of the strip's length survives (at this width the mesh resolves it only partly).
    double minX = 1e9, maxX = -1e9;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        if (f.phi[v] >= 0.0) continue;
        const std::optional<Vec2d> x = LocalLog(tri, center, tri.vertexPoint(v), 12);
        if (!x) continue;
        minX = std::min(minX, x->x());
        maxX = std::max(maxX, x->x());
    }
    CHECK(maxX - minX > 0.4 * (2.0 * L));  // observed 0.47
}

TEST(sign_field_no_springls_outside)
{
    const IntrinsicTriangulation tri = Mesh(3);
    const double h = tri.meanEdgeLength();
    const SignField f = ReconstructSignField(tri, SpringlSegments{}, {1.5 * h, 4.0 * h});
    CHECK(!f.hasInside);
    CHECK(f.leaked);
    CHECK(f.phi.size() == tri.vertexCount());
    for (double p : f.phi) CHECK(p > 0.0);
}

TEST(sign_field_reports_leak)
{
    const IntrinsicTriangulation tri = Mesh(4);
    const double h = tri.meanEdgeLength();
    const SpringlSegments closed = ClosedOutline(tri, 0, Star(0.5), 0.5 * h);
    const SpringlSegments s = ClosedOutline(tri, 0, Star(0.5), 0.5 * h, 4.0 * h);  // a gap 4 h wide
    const SignField f = ReconstructSignField(tri, s, {1.5 * h, 4.0 * h});
    CHECK(!f.hasInside);
    for (double p : f.phi) CHECK(p > 0.0);
    CHECK(f.leaked);
    // The same outline closed has an inside and no leak.
    const SignField c = ReconstructSignField(tri, closed, {1.5 * h, 4.0 * h});
    CHECK(c.hasInside);
    CHECK(!c.leaked);
}

TEST(sign_field_leak_with_thin_spike_still_reported)
{
    // A gapped outline plus a thin spike (a wall local maximum): phi has negative values, but the leak shows.
    const IntrinsicTriangulation tri = Mesh(4);
    const double h = tri.meanEdgeLength();
    SpringlSegments s = ClosedOutline(tri, 0, Star(0.5), 0.5 * h, 4.0 * h);
    const double w = 0.75 * h, L = 5.0 * h;
    // A closed sliver far from the star (the star spans 0.6; the sliver sits at distance 0.9 along -y).
    const Polyline sliver = {{-L, -0.9 - w}, {L, -0.9 - w}, {L, -0.9 + w}, {-L, -0.9 + w}};
    const SpringlSegments t = ClosedOutline(tri, 0, sliver, 0.5 * h);
    s.segments.insert(s.segments.end(), t.segments.begin(), t.segments.end());
    const SignField f = ReconstructSignField(tri, s, {1.5 * h, 4.0 * h});
    CHECK(f.hasInside);
    CHECK(f.leaked);
}

TEST(sign_field_star_full_amplitude)
{
    const IntrinsicTriangulation tri = Mesh(5);
    const double h = tri.meanEdgeLength(), r = 0.6;
    const SpringlSegments s = ClosedOutline(tri, 0, Star(r, 200, 0.35), 0.5 * h);
    const SignField f = ReconstructSignField(tri, s, {1.5 * h, 4.0 * h});
    CHECK(f.hasInside);
    CHECK(!f.leaked);
    const SurfacePoint center{0, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    int wrong = 0, checked = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        if (ExactDistance(tri, s, tri.vertexPoint(v), 6, 1e9) < 1.5 * h) continue;
        const std::optional<Vec2d> x = LocalLog(tri, center, tri.vertexPoint(v), 12);
        if (!x) continue;
        checked++;
        wrong += InsideStar(*x, r, 0.35) != (f.phi[v] < 0);
    }
    CHECK(checked > 200);
    CHECK(wrong == 0);
    CHECK(WorstCrossing(tri, s, f.phi) <= 0.35 * h);
}

TEST(sign_field_magnitude_is_distance)
{
    const IntrinsicTriangulation tri = Mesh(4);
    const double h = tri.meanEdgeLength();
    const SpringlSegments s = ClosedOutline(tri, 0, Star(0.5), 0.5 * h);
    const double clamp = 4.0 * h;
    const SignField f = ReconstructSignField(tri, s, {1.5 * h, clamp});
    double worst = 0.0;
    int n = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const double d = ExactDistance(tri, s, tri.vertexPoint(v), 6, 1e9);
        if (d > 2.0 * h) continue;
        worst = std::max(worst, std::abs(std::abs(f.phi[v]) - d));
        n++;
    }
    CHECK(n > 100);
    CHECK(worst <= 0.05 * h);
}

TEST(sign_field_capped_march_matches_full)
{
    // Marching only to the band (plus a margin) leaves the far vertices unreached (U = +inf) on both
    // sides of the springls; the classification still finds the inside, and phi is unchanged.
    const IntrinsicTriangulation tri = Mesh(5);
    const double h = tri.meanEdgeLength(), r = 0.6;
    const SpringlSegments s = ClosedOutline(tri, 0, Star(r), 0.5 * h);
    SignFieldParams full{1.5 * h, 4.0 * h};
    SignFieldParams capped = full;
    capped.march = full.clamp + 0.6 * 1.2 * h;
    const std::vector<double> U = SpringlDistance(tri, s, full), Uc = SpringlDistance(tri, s, capped);
    size_t unreachedInside = 0, unreachedOutside = 0;
    double worstU = 0.0;
    const SignField a = SignFieldFromDistance(tri, U, full), b = SignFieldFromDistance(tri, Uc, capped);
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        if (std::isfinite(Uc[v])) {
            worstU = std::max(worstU, std::abs(Uc[v] - U[v]));
        } else {
            CHECK(U[v] > capped.march - 1e-12);
            (a.phi[v] < 0.0 ? unreachedInside : unreachedOutside)++;
        }
        CHECK(U[v] > capped.march || std::isfinite(Uc[v]));
    }
    CHECK(worstU == 0.0);
    CHECK(unreachedInside > 100 && unreachedOutside > 1000);
    CHECK(!b.leaked && b.hasInside);
    CHECK(a.phi == b.phi);
}

TEST(sign_field_side_keeps_exterior_pocket)
{
    // A square with an exterior chamber behind a mouth two capsules wide: the walls close the mouth, so
    // topology alone fills the chamber in; the springls' side keeps it outside.
    const IntrinsicTriangulation tri = Mesh(5);
    const double h = tri.meanEdgeLength();
    Polyline square = {{-6, -6}, {-1.5, -6}, {-1.5, -4}, {-3, -4}, {-3, 1}, {3, 1}, {3, -4}, {1.5, -4}, {1.5, -6}, {6, -6}, {6, 6}, {-6, 6}};
    for (Vec2d &p : square) p *= h;
    const SpringlSegments s = ClosedOutline(tri, 0, square, 0.5 * h);
    const SignFieldParams params{1.5 * h, 4.0 * h};
    const std::vector<double> U = SpringlDistance(tri, s, params);
    const SpringlSide side = NearestSide(tri, s);
    SignFieldParams corrected = params;
    corrected.correction = 1.0 * h;
    const SignField topology = SignFieldFromDistance(tri, U, params), oriented = SignFieldFromDistance(tri, U, corrected, side);
    // The chamber's center and a point of the square's body.
    const SurfacePoint center{0, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)};
    const auto vertexNear = [&](const Vec2d &x) {
        const SurfacePoint q = Trace(tri, center, x * h).end;
        int c = 0;
        for (int k = 1; k < 3; k++) c = (q.bary[k] > q.bary[c]) ? k : c;
        return tri.tail(3 * q.face + c);
    };
    const uint32_t chamber = vertexNear(Vec2d(0.0, -1.5)), body = vertexNear(Vec2d(-4.0, 3.0));
    std::printf("  chamber phi: topology %.3f h, with side %.3f h; body %.3f h\n", topology.phi[chamber] / h, oriented.phi[chamber] / h,
                oriented.phi[body] / h);
    CHECK(topology.phi[chamber] < 0.0);  // the closed mouth: the chamber filled in
    CHECK(oriented.phi[chamber] > 0.0);
    CHECK(oriented.phi[body] < 0.0 && topology.phi[body] < 0.0);
    CHECK(!oriented.leaked);
    // Every sign change of the oriented field lies near the outline.
    int count = 0;
    const double worst = WorstCrossing(tri, s, oriented.phi, &count);
    std::printf("  %d crossings, worst %.3f h from the outline (topology alone: %.3f h)\n", count, worst / h, WorstCrossing(tri, s, topology.phi) / h);
    CHECK(worst < 0.5 * h);
}

TEST(sign_field_side_inside_larger_than_half)
{
    // A circle of geodesic radius 0.6 on the unit sphere, clockwise: its left, the inside, is the far side,
    // 88% of the sphere. The exterior flood starts from the largest U (the antipode, inside) and so floods
    // the inside; the springls' side must swap it.
    const IntrinsicTriangulation tri = Mesh(5);
    const double h = tri.meanEdgeLength(), rho = 0.6;
    Polyline circle;
    for (int i = 0; i < 400; i++) {
        const double t = -2.0 * std::numbers::pi * i / 400;
        circle.push_back(rho * Vec2d(std::cos(t), std::sin(t)));
    }
    const SpringlSegments s = ClosedOutline(tri, 0, circle, 0.5 * h);
    SignFieldParams params{1.5 * h, 4.0 * h};
    params.correction = 1.0 * h;
    const SignField f = SignFieldFromDistance(tri, SpringlDistance(tri, s, params), params, NearestSide(tri, s));
    const Vec3d c = tri.toExtrinsic(SurfacePoint{0, Vec3d(1.0 / 3, 1.0 / 3, 1.0 / 3)}).position.normalized();
    int wrong = 0, inside = 0, outside = 0;
    for (uint32_t v = 0; v < tri.vertexCount(); v++) {
        const double d = std::acos(std::clamp(tri.vertexPosition(v).normalized().dot(c), -1.0, 1.0));
        if (std::abs(d - rho) < h) continue;
        const bool in = d > rho;
        (in ? inside : outside)++;
        wrong += in != (f.phi[v] < 0.0);
    }
    std::printf("  %d inside, %d outside, %d wrong, leaked %d\n", inside, outside, wrong, int(f.leaked));
    CHECK(inside > 4 * outside);
    CHECK(wrong == 0);
    CHECK(!f.leaked && f.hasInside);
}
