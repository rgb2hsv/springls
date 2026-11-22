#include <intrinsic/IntrinsicDistance.hpp>

#include <core/MutableHeap.hpp>
#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace springls
{
namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();
// Point sources: vertices within this many mean edge lengths start from exact unfolded distances.
constexpr double kPointSourceRadius = 3.0;
constexpr int kPointSourceChartDepth = 10;
// Fronts farther than this many mean edge lengths from their point source advance as planes: the
// circle model's correction is small there, and on a curved surface far fronts are no longer circles
// about the unfolded source (on a sphere they converge again toward the antipode).
constexpr double kCircleFrontRadius = 6.0;

double Cross(const Vec2d &a, const Vec2d &b) { return a.x() * b.y() - a.y() * b.x(); }

// A front value with the radius of its front: the distance to the front's virtual point source
// (+inf for a front started from segments, which advances as a plane).
struct FrontValue {
    double t = kInf;
    double radius = kInf;
};
bool Better(const FrontValue &a, const FrontValue &b) { return a.t < b.t; }

// Triangle update of the value at pc from known values at pa and pb (c not on line ab). A front with
// finite radii is a circle about the virtual source S (|S - a| = ra, |S - b| = rb, on the far side of
// ab from c), which is exact for point sources; otherwise the front through a and b advances as a
// plane. Invalid (t = +inf) unless the characteristic through c comes through the segment ab.
FrontValue TriangleUpdate(const Vec2d &pa, const FrontValue &a, const Vec2d &pb, const FrontValue &b, const Vec2d &pc)
{
    const Vec2d e = pb - pa;
    const double L = e.norm();
    if (!(L > 0.0)) return {};
    const Vec2d ex = e / L;
    Vec2d ey(-ex.y(), ex.x());
    if (Cross(e, pc - pa) < 0.0) ey = -ey;  // toward c
    const Vec2d w = pc - pa;
    const double height = w.dot(ey);
    if (!(height > 0.0)) return {};
    // Where a ray into c along `dir` from behind meets line ab, as a fraction of ab.
    const auto crossesBase = [&](const Vec2d &from) {
        const Vec2d d = pc - from;
        const double dy = d.dot(ey);
        if (!(dy > 0.0)) return false;
        const double s = (pc - pa).dot(ey) / dy;  // back from c to the line
        const double t = (w - s * d).dot(ex) / L;
        return t >= 0.0 && t <= 1.0;
    };
    if (std::isfinite(a.radius) && std::isfinite(b.radius) && std::abs(a.radius - b.radius) <= L && a.radius + b.radius >= L) {
        const double x = (a.radius * a.radius + L * L - b.radius * b.radius) / (2.0 * L);
        const Vec2d S = pa + x * ex - std::sqrt(std::max(0.0, a.radius * a.radius - x * x)) * ey;
        if (!crossesBase(S)) return {};
        const double offset = 0.5 * ((a.t - a.radius) + (b.t - b.radius));
        const double r = (pc - S).norm();
        return {offset + r, r};
    }
    if (std::abs(b.t - a.t) > L) return {};
    const double nx = (b.t - a.t) / L;
    const Vec2d n = nx * ex + std::sqrt(std::max(0.0, 1.0 - nx * nx)) * ey;
    if (!crossesBase(pc - n)) return {};
    return {a.t + n.dot(w), kInf};
}

FrontValue EdgeUpdate(const FrontValue &a, double length) { return {a.t + length, a.radius + length}; }

Vec2d PlaceBelow(double base, double fromOrigin, double fromEnd)
{
    const double x = (fromOrigin * fromOrigin + base * base - fromEnd * fromEnd) / (2.0 * base);
    return Vec2d(x, -std::sqrt(std::max(0.0, fromOrigin * fromOrigin - x * x)));
}

struct Marcher {
    const IntrinsicTriangulation &tri;
    const std::vector<FrontValue> &F;
    const std::vector<uint8_t> &known;
    double maxRadius;

    // Best value at tail(he)'s vertex c from the face of outgoing halfedge he (corners c, a, b).
    FrontValue faceUpdate(uint32_t he) const
    {
        const uint32_t ha = tri.next(he), hb = tri.prev(he);
        const uint32_t a = tri.tail(ha), b = tri.tail(hb);
        const bool ka = known[a], kb = known[b];
        if (!ka && !kb) return {};
        const double lca = tri.halfedgeLength(he), lab = tri.halfedgeLength(ha), lbc = tri.halfedgeLength(hb);
        FrontValue best;
        const auto consider = [&](const FrontValue &v) {
            if (Better(v, best)) best = v;
        };
        if (ka) consider(EdgeUpdate(F[a], lca));
        if (kb) consider(EdgeUpdate(F[b], lbc));
        if (!(ka && kb)) return best;
        // Layout with a at the origin, b on +x, c above.
        const Vec2d pa(0, 0), pb(lab, 0);
        const Vec2d below = PlaceBelow(lab, lca, lbc);
        const Vec2d pc(below.x(), -below.y());
        const FrontValue direct = TriangleUpdate(pa, F[a], pb, F[b], pc);
        if (std::isfinite(direct.t)) {
            consider(direct);
            return best;
        }
        // Obtuse corner at c: unfold the face across ab and split the update at its far vertex u.
        const uint32_t tw = tri.twin(ha);  // b→a in the opposite face
        const uint32_t u = tri.tail(tri.prev(tw));
        if (!known[u]) return best;
        const double lau = tri.halfedgeLength(tri.next(tw)), lub = tri.halfedgeLength(tri.prev(tw));
        const Vec2d pu = PlaceBelow(lab, lau, lub);
        consider(TriangleUpdate(pa, F[a], pu, F[u], pc));
        consider(TriangleUpdate(pu, F[u], pb, F[b], pc));
        return best;
    }

    FrontValue vertexUpdate(uint32_t v) const
    {
        FrontValue best;
        const uint32_t start = tri.vertexHalfedge(v);
        uint32_t he = start;
        do {
            const FrontValue candidate = faceUpdate(he);
            if (Better(candidate, best)) best = candidate;
            he = tri.nextOutgoing(he);
        } while (he != start);
        if (best.radius > maxRadius) best.radius = kInf;
        return best;
    }
};
}  // namespace

namespace
{
// Per-thread buffers of FastMarch, sized to the largest mesh seen and reset after each call over the
// vertices it touched only, so a call costs what its front visits (plus the dense result).
struct MarchWorkspace {
    std::vector<FrontValue> F;
    std::vector<uint8_t> known, fixed, preset, touched;
    std::vector<uint32_t> list;  // vertices with touched[v]
    DenseMutableHeap<double> heap;

    void prepare(size_t V)
    {
        if (F.size() < V) {
            F.resize(V);
            known.resize(V, 0);
            fixed.resize(V, 0);
            preset.resize(V, 0);
            touched.resize(V, 0);
            heap.reserveKeys(V);
        }
    }
    void touch(uint32_t v)
    {
        if (!touched[v]) {
            touched[v] = 1;
            list.push_back(v);
        }
    }
    void reset()
    {
        for (uint32_t v : list) {
            F[v] = FrontValue{};
            known[v] = fixed[v] = preset[v] = touched[v] = 0;
        }
        list.clear();
        heap.clear();
    }
};
}  // namespace

std::vector<double> FastMarch(const IntrinsicTriangulation &tri, const std::vector<DistanceSeed> &seeds, double maxDistance)
{
    const size_t V = tri.vertexCount();
    thread_local MarchWorkspace ws;
    ws.prepare(V);
    std::vector<FrontValue> &F = ws.F;
    std::vector<uint8_t> &known = ws.known, &fixed = ws.fixed, &preset = ws.preset;
    for (const DistanceSeed &s : seeds) {
        ws.touch(s.vertex);
        F[s.vertex].t = std::min(F[s.vertex].t, s.distance);
        fixed[s.vertex] = 1;
    }
    // Seeds none of which touches another are point sources: each front is a circle (radius 0 at
    // the source), and each source's neighborhood starts from exact distances in an unfolding
    // (shortest over the unfoldings of each vertex), where a first-order update is least accurate.
    // These are starting values, not seeds: a closer seed still lowers them. Seeds that touch (a
    // sampled front, e.g. vertices beside a curve) march as fronts throughout.
    const double h = tri.meanEdgeLength();
    bool pointSources = true;
    for (const DistanceSeed &s : seeds) {
        const uint32_t start = tri.vertexHalfedge(s.vertex);
        uint32_t he = start;
        do {
            pointSources &= !fixed[tri.head(he)];
            he = tri.nextOutgoing(he);
        } while (he != start && pointSources);
        if (!pointSources) break;
    }
    std::vector<std::pair<uint32_t, FrontValue>> exact;
    for (const DistanceSeed &s : seeds) {
        if (!pointSources) break;
        F[s.vertex].radius = 0.0;
        const SurfacePoint source = tri.vertexPoint(s.vertex);
        const LocalChart chart(tri, source.face, kPointSourceChartDepth, 3);
        const Vec2d origin = chart.position(source);
        for (uint32_t f : chart.faces()) {
            for (const auto &C : chart.placements(f)) {
                for (int c = 0; c < 3; c++) {
                    const double d = (C[c] - origin).norm();
                    if (d <= kPointSourceRadius * h) exact.emplace_back(tri.tail(3 * f + c), FrontValue{F[s.vertex].t + d, d});
                }
            }
        }
    }
    for (const auto &[v, value] : exact) {
        if (fixed[v]) continue;  // given seeds are never changed
        ws.touch(v);
        if (!preset[v] || Better(value, F[v])) F[v] = value;
        preset[v] = 1;
    }
    // Preset values are tentative: marching from other seeds may still lower them. One heap entry per
    // vertex, lowered in place; popped smallest distance first, ties by the smaller vertex index.
    DenseMutableHeap<double> &heap = ws.heap;
    for (uint32_t v : ws.list) heap.push(v, F[v].t);
    const Marcher marcher{tri, F, known, kCircleFrontRadius * h};
    while (!heap.empty()) {
        const auto [v, d] = *heap.top();
        if (d > maxDistance) break;
        heap.pop();
        known[v] = 1;
        const uint32_t start = tri.vertexHalfedge(v);
        uint32_t he = start;
        do {
            const uint32_t w = tri.head(he);
            if (!known[w] && !fixed[w]) {
                const FrontValue value = marcher.vertexUpdate(w);
                if (Better(value, F[w])) {
                    ws.touch(w);
                    F[w] = value;
                    if (!heap.exchange(w, value.t)) heap.push(w, value.t);
                }
            }
            he = tri.nextOutgoing(he);
        } while (he != start);
    }
    // Given seeds keep their values; anything else never accepted (beyond the stop distance) is dropped.
    std::vector<double> D(V, kInf);
    for (const DistanceSeed &s : seeds) D[s.vertex] = F[s.vertex].t;
    for (uint32_t v : ws.list) {
        if (known[v]) D[v] = F[v].t;
    }
    ws.reset();
    return D;
}

std::vector<DistanceSeed> SeedsFromSegments(const IntrinsicTriangulation &tri, const std::vector<std::pair<SurfacePoint, SurfacePoint>> &segments)
{
    std::map<uint32_t, double> best;
    const auto seedFace = [&](uint32_t f, const Vec2d &a, const Vec2d &b) {
        const auto P = tri.layout(f);
        const Vec2d ab = b - a;
        const double len2 = ab.squaredNorm();
        for (int c = 0; c < 3; c++) {
            const double t = (len2 > 0.0) ? std::clamp((P[c] - a).dot(ab) / len2, 0.0, 1.0) : 0.0;
            const double d = (P[c] - (a + t * ab)).norm();
            const uint32_t v = tri.tail(3 * f + c);
            const auto it = best.find(v);
            if (it == best.end() || d < it->second) best[v] = d;
        }
    };
    for (const auto &[p, q] : segments) {
        const Vec2d pp = tri.toLayout(p), qq = tri.toLayout(q);
        if (p.face == q.face) {
            seedFace(p.face, pp, qq);
            continue;
        }
        const auto pq = LocalLog(tri, p, q), qp = LocalLog(tri, q, p);
        seedFace(p.face, pp, pq ? Vec2d(pp + *pq) : pp);
        seedFace(q.face, qp ? Vec2d(qq + *qp) : qq, qq);
    }
    std::vector<DistanceSeed> seeds;
    for (const auto &[v, d] : best) seeds.push_back({v, d});
    return seeds;
}

Vec2d FaceGradient(const IntrinsicTriangulation &tri, uint32_t face, const std::vector<double> &vertexValues)
{
    const auto P = tri.layout(face);
    const double A2 = Cross(P[1] - P[0], P[2] - P[0]);
    Vec2d g = Vec2d::Zero();
    for (int c = 0; c < 3; c++) {
        const Vec2d e = P[(c + 2) % 3] - P[(c + 1) % 3];
        g += vertexValues[tri.tail(3 * face + c)] * Vec2d(-e.y(), e.x()) / A2;
    }
    return g;
}
}  // namespace springls
