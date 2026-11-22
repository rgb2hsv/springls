#include <intrinsic/IntrinsicReference.hpp>

#include <intrinsic/IntrinsicDistance.hpp>
#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace springls
{
namespace
{
// The point of face f at layout position x (barycentric coordinates from the layout corners).
SurfacePoint FromLayout(const IntrinsicTriangulation &tri, uint32_t f, const Vec2d &x)
{
    const auto P = tri.layout(f);
    const Vec2d e1 = P[1] - P[0], e2 = P[2] - P[0], d = x - P[0];
    const double det = e1.x() * e2.y() - e1.y() * e2.x();
    const double b1 = (d.x() * e2.y() - d.y() * e2.x()) / det, b2 = (e1.x() * d.y() - e1.y() * d.x()) / det;
    SurfacePoint p;
    p.face = f;
    p.bary = Vec3d(1.0 - b1 - b2, b1, b2);
    return p;
}
}  // namespace

void IntrinsicReference::initialize(const IntrinsicTriangulation &tri, const std::vector<IntrinsicCurve> &contour, int subdivisions)
{
    mCurves.clear();
    mLabels.clear();
    mClosed.clear();
    subdivisions = std::max(1, subdivisions);
    for (const IntrinsicCurve &curve : contour) {
        const size_t n = curve.points.size();
        if (n == 0) continue;
        std::vector<SurfacePoint> points;
        const size_t segments = curve.closed ? n : n - 1;
        for (size_t k = 0; k < segments; k++) {
            const uint32_t f = curve.faces[k];
            const SurfacePoint a = ContourPointInFace(tri, curve, k, f), b = ContourPointInFace(tri, curve, (k + 1) % n, f);
            const Vec2d pa = tri.toLayout(a), pb = tri.toLayout(b);
            points.push_back(a);
            for (int s = 1; s < subdivisions; s++) points.push_back(FromLayout(tri, f, pa + (pb - pa) * (static_cast<double>(s) / subdivisions)));
        }
        if (!curve.closed) points.push_back(curve.points.back());
        mCurves.push_back(std::move(points));
        mLabels.push_back(curve.label);
        mClosed.push_back(curve.closed);
    }
}

std::vector<SurfacePoint *> IntrinsicReference::points()
{
    std::vector<SurfacePoint *> out;
    for (auto &curve : mCurves) {
        for (SurfacePoint &p : curve) out.push_back(&p);
    }
    return out;
}

void IntrinsicReference::advance(const IntrinsicTriangulation &tri, const std::function<Vec2d(const SurfacePoint &, double)> &velocity, double fromTime,
                                 double toTime, double maxStep, const std::vector<double> &breaks)
{
    constexpr double kTiny = 1e-12;
    double t = fromTime;
    // Every point, flattened for the parallel loops (each point moves on its own).
    std::vector<SurfacePoint *> all;
    for (auto &curve : mCurves) {
        for (SurfacePoint &p : curve) all.push_back(&p);
    }
    const int n = static_cast<int>(all.size());
    while (t < toTime - kTiny * std::max(1.0, std::abs(toTime))) {
        double maxSpeed = 0.0;
        SPRINGLS_OMP(omp parallel for reduction(max : maxSpeed) if (n >= kParallelMinHeavyItems))
        for (int i = 0; i < n; i++) maxSpeed = std::max(maxSpeed, velocity(*all[i], t).norm());
        double dt = toTime - t;
        if (maxSpeed > 0.0) dt = std::min(dt, maxStep / maxSpeed);
        bool landsOnBreak = false;
        for (double b : breaks) {
            if (b > t + kTiny && b <= t + dt) {
                dt = b - t;
                landsOnBreak = true;
            }
        }
        const double tEnd = (t + dt >= toTime - kTiny) ? toTime : t + dt;
        const double tHalf = t + 0.5 * dt;
        // The last stage sees the flow as it is just before a break, not after it.
        const double tLast = landsOnBreak ? std::nextafter(tEnd, t) : tEnd;
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 16) if (n >= kParallelMinHeavyItems))
        for (int i = 0; i < n; i++) {
            SurfacePoint &x = *all[i];
            const Vec2d v1 = velocity(x, t);
            const TraceResult r2 = Trace(tri, x, (0.5 * dt) * v1);
            const Vec2d v2 = TransportBack(r2, velocity(r2.end, tHalf));
            const TraceResult r3 = Trace(tri, x, (0.5 * dt) * v2);
            const Vec2d v3 = TransportBack(r3, velocity(r3.end, tHalf));
            const TraceResult r4 = Trace(tri, x, dt * v3);
            const Vec2d v4 = TransportBack(r4, velocity(r4.end, tLast));
            x = Trace(tri, x, (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)).end;
        }
        t = tEnd;
    }
}

double IntrinsicReference::maxGap(const IntrinsicTriangulation &tri) const
{
    // Every segment (curve, start index), then their log maps in parallel.
    std::vector<std::pair<uint32_t, uint32_t>> segments;
    for (size_t c = 0; c < mCurves.size(); c++) {
        const size_t count = mClosed[c] ? mCurves[c].size() : (mCurves[c].empty() ? 0 : mCurves[c].size() - 1);
        for (size_t k = 0; k < count; k++) segments.emplace_back(static_cast<uint32_t>(c), static_cast<uint32_t>(k));
    }
    const int n = static_cast<int>(segments.size());
    double gap = 0.0;
    int unreached = 0;
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 8) reduction(max : gap) reduction(+ : unreached) if (n >= kParallelMinHeavyItems))
    for (int i = 0; i < n; i++) {
        const auto &pts = mCurves[segments[i].first];
        const size_t k = segments[i].second;
        const SurfacePoint &a = pts[k], &b = pts[(k + 1) % pts.size()];
        if (a.face == b.face) {
            // In one face the straight segment is the geodesic: its length in the face's layout.
            gap = std::max(gap, (tri.toLayout(b) - tri.toLayout(a)).norm());
            continue;
        }
        const auto log = LocalLog(tri, a, b);
        if (!log) {
            unreached++;
            continue;
        }
        gap = std::max(gap, log->norm());
    }
    return unreached > 0 ? std::numeric_limits<double>::infinity() : gap;
}

std::vector<double> IntrinsicReference::distanceField(const IntrinsicTriangulation &tri, double maxDistance) const
{
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;
    for (size_t c = 0; c < mCurves.size(); c++) {
        const auto &pts = mCurves[c];
        const size_t count = mClosed[c] ? pts.size() : (pts.empty() ? 0 : pts.size() - 1);
        for (size_t k = 0; k < count; k++) segments.emplace_back(pts[k], pts[(k + 1) % pts.size()]);
        if (pts.size() == 1) segments.emplace_back(pts[0], pts[0]);
    }
    return FastMarch(tri, SeedsFromSegments(tri, segments), maxDistance);
}

std::vector<double> IntrinsicReference::distancesAt(const IntrinsicTriangulation &tri, const std::vector<SurfacePoint> &points, double cap) const
{
    constexpr int kChartDepth = 6;  // as the springls' charts
    // Reference points by face, as (curve, index).
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> byFace;
    for (uint32_t c = 0; c < mCurves.size(); c++) {
        for (uint32_t k = 0; k < mCurves[c].size(); k++) byFace[mCurves[c][k].face].emplace_back(c, k);
    }
    const auto segmentCount = [&](size_t c) { return mClosed[c] ? mCurves[c].size() : (mCurves[c].empty() ? 0 : mCurves[c].size() - 1); };
    // Charts of the points' faces, built in parallel.
    std::vector<uint32_t> faces;
    faces.reserve(points.size());
    for (const SurfacePoint &p : points) faces.push_back(p.face);
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    std::vector<LocalChart> charts(faces.size());
    const int F = static_cast<int>(faces.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (F >= 16))
    for (int f = 0; f < F; f++) charts[f].build(tri, faces[f], kChartDepth);
    const auto chartOf = [&](uint32_t face) -> const LocalChart & { return charts[std::lower_bound(faces.begin(), faces.end(), face) - faces.begin()]; };
    // Exact distances where a reference segment lies in the point's chart; +inf marks the others.
    const int n = static_cast<int>(points.size());
    std::vector<double> result(points.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 16) if (n >= kParallelMinHeavyItems))
    for (int i = 0; i < n; i++) {
        const SurfacePoint &p = points[i];
        const LocalChart &chart = chartOf(p.face);
        std::vector<std::pair<uint32_t, uint32_t>> segments;  // (curve, start index)
        for (uint32_t f : chart.faces()) {
            const auto found = byFace.find(f);
            if (found == byFace.end()) continue;
            for (const auto &[c, k] : found->second) {
                segments.emplace_back(c, k);
                const size_t m = mCurves[c].size();
                if (mClosed[c] || k > 0) segments.emplace_back(c, static_cast<uint32_t>((k + m - 1) % m));
            }
        }
        std::sort(segments.begin(), segments.end());
        segments.erase(std::unique(segments.begin(), segments.end()), segments.end());
        const Vec2d x = chart.position(p);
        double best = std::numeric_limits<double>::infinity();
        for (const auto &[c, k] : segments) {
            if (k >= segmentCount(c)) continue;
            const SurfacePoint &a = mCurves[c][k], &b = mCurves[c][(k + 1) % mCurves[c].size()];
            if (!chart.contains(a.face) || !chart.contains(b.face)) continue;
            const Vec2d pa = chart.position(a), pb = chart.position(b), ab = pb - pa;
            const double len2 = ab.squaredNorm();
            const double t = (len2 > 0.0) ? std::clamp((x - pa).dot(ab) / len2, 0.0, 1.0) : 0.0;
            best = std::min(best, (x - (pa + t * ab)).norm());
        }
        result[i] = best;
    }
    // The fast-marching field (only if some point needs it) for the rest; then the cap.
    std::vector<double> field;
    for (int i = 0; i < n; i++) {
        double best = result[i];
        if (!std::isfinite(best)) {
            if (field.empty()) field = distanceField(tri, cap);
            const SurfacePoint &p = points[i];
            best = 0.0;
            for (int c = 0; c < 3; c++) best += p.bary[c] * field[tri.tail(3 * p.face + c)];
        }
        result[i] = std::isfinite(best) ? std::min(best, cap) : cap;
    }
    return result;
}
}  // namespace springls
