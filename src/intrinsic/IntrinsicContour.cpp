#include <intrinsic/IntrinsicContour.hpp>

#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace springls
{
namespace
{
double Cross(const Vec2d &a, const Vec2d &b) { return a.x() * b.y() - a.y() * b.x(); }

// Parameter on the canonical halfedge of edge e where the linear interpolant crosses zero.
double CrossingParameter(const IntrinsicTriangulation &tri, uint32_t e, const std::function<double(uint32_t)> &value)
{
    const uint32_t he = tri.edgeHalfedge(e);
    const double a = value(tri.tail(he)), b = value(tri.head(he));
    const double denom = a - b;
    return (denom != 0.0) ? std::clamp(a / denom, 0.0, 1.0) : 0.5;
}
}  // namespace

SurfacePoint ContourPointInFace(const IntrinsicTriangulation &tri, const IntrinsicCurve &curve, size_t k, uint32_t f)
{
    const uint32_t he = tri.edgeHalfedge(curve.edges[k]);
    if (tri.face(he) == f) return IntrinsicTriangulation::halfedgePoint(he, curve.t[k]);
    return IntrinsicTriangulation::halfedgePoint(tri.twin(he), 1.0 - curve.t[k]);
}

std::vector<IntrinsicCurve> ExtractIntrinsicContour(const IntrinsicTriangulation &tri, const std::function<double(uint32_t, int)> &signedValue,
                                                    const std::vector<int> &objectLabels)
{
    std::vector<IntrinsicCurve> curves;
    const uint32_t F = static_cast<uint32_t>(tri.faceCount());
    for (int label : objectLabels) {
        const auto value = [&](uint32_t v) { return signedValue(v, label); };
        // Per face with a crossing: the segment from the exit edge (tail inside, head outside) to the
        // entry edge (tail outside, head inside), which keeps the region on the left.
        struct Segment {
            uint32_t face, from, to;  // edges
        };
        std::vector<Segment> segments;
        std::unordered_map<uint32_t, size_t> startingAt;  // exit edge -> segment
        for (uint32_t f = 0; f < F; f++) {
            bool inside[3];
            for (int c = 0; c < 3; c++) inside[c] = value(tri.tail(3 * f + c)) < 0.0;
            if (inside[0] == inside[1] && inside[1] == inside[2]) continue;
            uint32_t exitEdge = 0, entryEdge = 0;
            for (int c = 0; c < 3; c++) {
                const bool a = inside[c], b = inside[(c + 1) % 3];
                if (a && !b) exitEdge = tri.edge(3 * f + c);
                if (!a && b) entryEdge = tri.edge(3 * f + c);
            }
            startingAt[exitEdge] = segments.size();
            segments.push_back({f, exitEdge, entryEdge});
        }
        std::vector<uint8_t> used(segments.size(), 0);
        for (size_t s0 = 0; s0 < segments.size(); s0++) {
            if (used[s0]) continue;
            IntrinsicCurve curve;
            curve.label = label;
            size_t s = s0;
            while (true) {
                used[s] = 1;
                const Segment &seg = segments[s];
                const double t = CrossingParameter(tri, seg.from, value);
                curve.edges.push_back(seg.from);
                curve.t.push_back(t);
                curve.points.push_back(IntrinsicTriangulation::halfedgePoint(tri.edgeHalfedge(seg.from), t));
                curve.faces.push_back(seg.face);
                const auto next = startingAt.find(seg.to);
                if (next == startingAt.end()) {
                    // Open end (cannot happen on a closed surface): finish with the entry crossing.
                    const double te = CrossingParameter(tri, seg.to, value);
                    curve.edges.push_back(seg.to);
                    curve.t.push_back(te);
                    curve.points.push_back(IntrinsicTriangulation::halfedgePoint(tri.edgeHalfedge(seg.to), te));
                    break;
                }
                if (next->second == s0) {
                    curve.closed = true;
                    break;
                }
                if (used[next->second]) break;
                s = next->second;
            }
            curves.push_back(std::move(curve));
        }
    }
    return curves;
}

void RefineIntrinsicContour(const IntrinsicTriangulation &tri, std::vector<IntrinsicCurve> &curves, const std::vector<SurfacePoint> &particles,
                            const std::vector<int> &particleLabels, int iterations, double proximity, double stepSize)
{
    std::unordered_map<uint32_t, std::vector<uint32_t>> byFace;
    for (uint32_t n = 0; n < particles.size(); n++) byFace[particles[n].face].push_back(n);
    // Faces within reach of an edge's chart: enough rings for the proximity.
    const int depth = std::max(2, static_cast<int>(std::ceil(2.0 * proximity / tri.meanEdgeLength())) + 1);
    constexpr double kEdgeMargin = 1e-6;
    // Every crossing (curve, index) and the charts of their edges' faces, built once (in parallel) for all
    // iterations; each crossing then slides on its own (in parallel).
    std::vector<std::pair<uint32_t, uint32_t>> crossings;
    std::vector<uint32_t> faces;
    for (uint32_t c = 0; c < curves.size(); c++) {
        for (uint32_t k = 0; k < curves[c].points.size(); k++) {
            crossings.emplace_back(c, k);
            faces.push_back(tri.face(tri.edgeHalfedge(curves[c].edges[k])));
        }
    }
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    std::vector<LocalChart> charts(faces.size());
    const int F = static_cast<int>(faces.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (F >= 16))
    for (int i = 0; i < F; i++) charts[i].build(tri, faces[i], depth);
    const auto chartOf = [&](uint32_t face) -> const LocalChart & { return charts[std::lower_bound(faces.begin(), faces.end(), face) - faces.begin()]; };
    const int n = static_cast<int>(crossings.size());
    for (int iter = 0; iter < iterations; iter++) {
        SPRINGLS_OMP(omp parallel for schedule(dynamic, 64) if (n >= kParallelMinHeavyItems))
        for (int i = 0; i < n; i++) {
            IntrinsicCurve &curve = curves[crossings[i].first];
            const size_t k = crossings[i].second;
            const uint32_t he = tri.edgeHalfedge(curve.edges[k]);
            const LocalChart &chart = chartOf(tri.face(he));
            const auto P = tri.layout(tri.face(he));
            const int c = static_cast<int>(he % 3);
            const Vec2d A = P[c], B = P[(c + 1) % 3];
            const Vec2d x = A + curve.t[k] * (B - A);
            double best = proximity;
            Vec2d closest = x;
            for (uint32_t f : chart.faces()) {
                const auto it = byFace.find(f);
                if (it == byFace.end()) continue;
                for (uint32_t m : it->second) {
                    if (particleLabels[m] != curve.label) continue;
                    const Vec2d p = chart.position(particles[m]);
                    const double d = (p - x).norm();
                    if (d < best) {
                        best = d;
                        closest = p;
                    }
                }
            }
            if (best >= proximity) continue;
            const double target = std::clamp((closest - A).dot(B - A) / (B - A).squaredNorm(), 0.0, 1.0);
            const double w = 1.0 - best / proximity;
            const double t = std::clamp(curve.t[k] + stepSize * w * (target - curve.t[k]), kEdgeMargin, 1.0 - kEdgeMargin);
            curve.t[k] = t;
            curve.points[k] = IntrinsicTriangulation::halfedgePoint(he, t);
        }
    }
}

namespace
{
// Clips face f's layout to {φ < 0}: the inside corners and the crossings between them, in order. A crossing on
// an edge is at the linear interpolant's zero unless `crossing` gives its parameter along the face's halfedge.
// Returns the corner count n (0: outside; the face is wholly inside when every corner is).
int ClipFace(const IntrinsicTriangulation &tri, uint32_t f, const std::function<double(uint32_t)> &signedValue,
             const std::function<bool(uint32_t he, double &t)> &crossing, Vec2d (&poly)[4], int &inside)
{
    double s[3];
    inside = 0;
    for (int c = 0; c < 3; c++) {
        s[c] = signedValue(tri.tail(3 * f + c));
        inside += s[c] < 0.0;
    }
    if (inside == 0) return 0;
    const auto P = tri.layout(f);
    int n = 0;
    for (int c = 0; c < 3; c++) {
        const int d = (c + 1) % 3;
        if (s[c] < 0.0) poly[n++] = P[c];
        if ((s[c] < 0.0) != (s[d] < 0.0)) {
            double t = s[c] / (s[c] - s[d]);
            if (crossing) crossing(3 * f + c, t);
            poly[n++] = P[c] + t * (P[d] - P[c]);
        }
    }
    return n;
}

// Area of {φ < 0} per face (see ClipFace).
double ClippedArea(const IntrinsicTriangulation &tri, const std::function<double(uint32_t)> &signedValue,
                   const std::function<bool(uint32_t he, double &t)> &crossing)
{
    double area = 0.0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        Vec2d poly[4];
        int inside = 0;
        const int n = ClipFace(tri, f, signedValue, crossing, poly, inside);
        if (inside == 0) continue;
        if (inside == 3) {
            area += tri.faceArea(f);
            continue;
        }
        double a = 0.0;
        for (int k = 0; k < n; k++) a += Cross(poly[k], poly[(k + 1) % n]);
        area += 0.5 * std::abs(a);
    }
    return area;
}
}  // namespace

double RegionArea(const IntrinsicTriangulation &tri, const std::function<double(uint32_t)> &signedValue) { return ClippedArea(tri, signedValue, nullptr); }

std::vector<RegionPiece> RegionPieces(const IntrinsicTriangulation &tri, const std::function<double(uint32_t)> &signedValue)
{
    std::vector<RegionPiece> pieces;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        Vec2d poly[4];
        int inside = 0;
        const int n = ClipFace(tri, f, signedValue, nullptr, poly, inside);
        if (inside == 0) continue;
        RegionPiece piece;
        piece.face = f;
        piece.centroid.face = f;
        if (inside == 3) {
            piece.area = tri.faceArea(f);
            piece.centroid.bary = Vec3d(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
            pieces.push_back(piece);
            continue;
        }
        // The polygon's centroid (fan from its first corner), then its barycentric coordinates in the layout.
        double a = 0.0;
        Vec2d c = Vec2d::Zero();
        for (int k = 1; k + 1 < n; k++) {
            const double w = 0.5 * Cross(poly[k] - poly[0], poly[k + 1] - poly[0]);
            a += w;
            c += w * (poly[0] + poly[k] + poly[k + 1]) / 3.0;
        }
        if (!(std::abs(a) > 0.0)) continue;
        c /= a;
        const auto P = tri.layout(f);
        const double total = Cross(P[1] - P[0], P[2] - P[0]);
        const double b1 = Cross(c - P[0], P[2] - P[0]) / total, b2 = Cross(P[1] - P[0], c - P[0]) / total;
        piece.area = std::abs(a);
        piece.centroid.bary = Vec3d(1.0 - b1 - b2, b1, b2);
        pieces.push_back(piece);
    }
    return pieces;
}

double ContourArea(const IntrinsicTriangulation &tri, const std::vector<IntrinsicCurve> &curves, int label,
                   const std::function<double(uint32_t, int)> &signedValue)
{
    std::unordered_map<uint32_t, double> t;  // edge -> parameter on its canonical halfedge
    for (const IntrinsicCurve &c : curves) {
        if (c.label != label) continue;
        for (size_t k = 0; k < c.edges.size(); k++) t[c.edges[k]] = c.t[k];
    }
    return ClippedArea(
        tri, [&](uint32_t v) { return signedValue(v, label); },
        [&](uint32_t he, double &param) {
            const auto it = t.find(tri.edge(he));
            if (it == t.end()) return false;
            param = (tri.edgeHalfedge(tri.edge(he)) == he) ? it->second : 1.0 - it->second;
            return true;
        });
}
}  // namespace springls
