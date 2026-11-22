#include <intrinsic/IntrinsicTrace.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace springls
{
namespace
{
double Angle(const Vec2d &v) { return std::atan2(v.y(), v.x()); }
// Rotation by a unit complex number r = (cos, sin), and its inverse.
Vec2d Turn(const Vec2d &v, const Vec2d &r) { return Vec2d(r.x() * v.x() - r.y() * v.y(), r.y() * v.x() + r.x() * v.y()); }
Vec2d TurnBack(const Vec2d &v, const Vec2d &r) { return Vec2d(r.x() * v.x() + r.y() * v.y(), -r.y() * v.x() + r.x() * v.y()); }
}  // namespace

Vec2d ToLayout(const IntrinsicTriangulation &tri, const SurfacePoint &p) { return tri.toLayout(p); }

Vec2d Rotate(const Vec2d &v, double angle)
{
    const double c = std::cos(angle), s = std::sin(angle);
    return Vec2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

TraceResult Trace(const IntrinsicTriangulation &tri, const SurfacePoint &start, const Vec2d &displacement, std::vector<EdgeCrossing> *crossings)
{
    const IntrinsicTriangulation::Trace t = tri.trace(start, displacement, crossings);
    TraceResult r;
    r.end = t.end;
    r.direction = t.direction;
    r.rotation = Angle(t.direction) - Angle(t.startDirection);
    return r;
}

Vec2d TransportBack(const TraceResult &trace, const Vec2d &vectorAtEnd) { return Rotate(vectorAtEnd, -trace.rotation); }

void LocalChart::insertFirst(uint32_t face, uint32_t entry)
{
    // Keep the load at most a half.
    if (2 * (mFaces.size()) > mSlots.size()) {
        std::vector<uint32_t> old = std::move(mSlots);
        mSlots.assign(std::max<size_t>(16, 2 * old.size()), kNoEntry);
        const size_t mask = mSlots.size() - 1;
        for (uint32_t e : old) {
            if (e == kNoEntry) continue;
            size_t k = Hash(mFaces[e]) & mask;
            while (mSlots[k] != kNoEntry) k = (k + 1) & mask;
            mSlots[k] = e;
        }
    }
    const size_t mask = mSlots.size() - 1;
    size_t k = Hash(face) & mask;
    while (mSlots[k] != kNoEntry) k = (k + 1) & mask;
    mSlots[k] = entry;
}

uint32_t LocalChart::at(uint32_t face) const
{
    const uint32_t e = find(face);
    if (e == kNoEntry) throw std::out_of_range("LocalChart: face not in the chart");
    return e;
}

void LocalChart::build(const IntrinsicTriangulation &tri, uint32_t centerFace, int depth, int maxPlacements, uint32_t stopAt)
{
    mCenter = centerFace;
    mFaces.clear();
    mCorners.clear();
    mRotation.clear();
    mNext.clear();
    mSlots.assign(64, kNoEntry);
    const double tolerance = 1e-9 * tri.halfedgeLength(3 * centerFace);
    std::vector<uint32_t> last;  // with maxPlacements > 1: per first entry, its face's last placement
    // Adds a placement of f (false if f already has maxPlacements placements or this one coincides with one).
    const auto add = [&](uint32_t f, const std::array<Vec2d, 3> &corners, const Vec2d &rotation) -> bool {
        const uint32_t first = find(f);
        if (first != kNoEntry) {
            int count = 0;
            for (uint32_t e = first; e != kNoEntry; e = mNext[e]) {
                count++;
                const auto &C = mCorners[e];
                if ((C[0] - corners[0]).norm() < tolerance && (C[1] - corners[1]).norm() < tolerance && (C[2] - corners[2]).norm() < tolerance) return false;
            }
            if (count >= maxPlacements) return false;
        }
        const uint32_t entry = static_cast<uint32_t>(mFaces.size());
        mFaces.push_back(f);
        mCorners.push_back(corners);
        mRotation.push_back(rotation);
        mNext.push_back(kNoEntry);
        if (first == kNoEntry) {
            insertFirst(f, entry);
            if (maxPlacements > 1) {
                if (last.size() <= entry) last.resize(entry + 1, kNoEntry);
                last[entry] = entry;
            }
        } else {
            mNext[last[first]] = entry;
            last[first] = entry;
        }
        return true;
    };
    // Breadth first: (entry, depth) in insertion order.
    std::vector<std::pair<uint32_t, int>> queue;
    add(centerFace, tri.layout(centerFace), Vec2d(1.0, 0.0));
    if (centerFace == stopAt) return;
    queue.emplace_back(0, 0);
    for (size_t head = 0; head < queue.size(); head++) {
        const auto [entry, d] = queue[head];
        if (d >= depth) continue;
        const uint32_t f = mFaces[entry];
        const std::array<Vec2d, 3> known = mCorners[entry];
        for (int c = 0; c < 3; c++) {
            const uint32_t he = 3 * f + c;
            const uint32_t tw = tri.twin(he);
            const uint32_t g = tri.face(tw);
            if (maxPlacements <= 1 && find(g) != kNoEntry) continue;
            // tw runs from head(he) to tail(he): its layout endpoints map onto the known edge.
            const auto Q = tri.layout(g);
            const int ct = static_cast<int>(tw % 3);
            const Vec2d A = known[c], B = known[(c + 1) % 3];
            // The rotation taking g's edge direction onto the known edge's: unit vectors' complex quotient.
            const Vec2d u = (A - B).normalized(), w = (Q[(ct + 1) % 3] - Q[ct]).normalized();
            const Vec2d rotation(u.x() * w.x() + u.y() * w.y(), u.y() * w.x() - u.x() * w.y());
            const Vec2d t = B - Turn(Q[ct], rotation);
            std::array<Vec2d, 3> corners;
            for (int k = 0; k < 3; k++) corners[k] = Turn(Q[k], rotation) + t;
            if (add(g, corners, rotation)) {
                if (g == stopAt) return;
                queue.emplace_back(static_cast<uint32_t>(mFaces.size() - 1), d + 1);
            }
        }
    }
}

std::vector<std::array<Vec2d, 3>> LocalChart::placements(uint32_t face) const
{
    // In entry order (the order they were placed).
    std::vector<std::array<Vec2d, 3>> result;
    for (uint32_t e = find(face); e != kNoEntry; e = mNext[e]) result.push_back(mCorners[e]);
    return result;
}

Vec2d LocalChart::vector(uint32_t face, const Vec2d &w) const { return Turn(w, mRotation[at(face)]); }

Vec2d LocalChart::vectorInFace(uint32_t face, const Vec2d &w) const { return TurnBack(w, mRotation[at(face)]); }

std::optional<Vec2d> LocalLog(const IntrinsicTriangulation &tri, const SurfacePoint &from, const SurfacePoint &to, int maxRings)
{
    // Candidates from every unfolding of `to`'s face (around a curved vertex, geodesics from both
    // sides can reach it), each corrected by tracing until it lands on `to`; the shortest wins.
    constexpr int kMaxPlacements = 3;
    const LocalChart chart(tri, from.face, 2 * maxRings, kMaxPlacements);
    if (!chart.contains(to.face)) return std::nullopt;
    const Vec2d origin = chart.position(from);
    const double scale = tri.halfedgeLength(3 * from.face);
    std::optional<Vec2d> best;
    for (const auto &C : chart.placements(to.face)) {
        Vec2d v = to.bary[0] * C[0] + to.bary[1] * C[1] + to.bary[2] * C[2] - origin;
        constexpr int kRefinements = 8;
        bool landed = false;
        for (int iter = 0; iter < kRefinements; iter++) {
            if (v.norm() == 0.0) {
                landed = true;
                break;
            }
            const TraceResult r = Trace(tri, from, v);
            // Only `to`'s placement is needed: the unfolding stops once it is placed (the same placement).
            LocalChart near;
            near.build(tri, r.end.face, 2, 1, to.face);
            if (!near.contains(to.face)) break;
            const Vec2d error = near.position(to) - near.position(r.end);  // in r.end.face's layout
            if (error.norm() <= 1e-12 * scale) {
                landed = true;
                break;
            }
            v += TransportBack(r, error);
        }
        if (landed && (!best || v.norm() < best->norm())) best = v;
    }
    if (!best) {
        // No candidate converged: the first unfolding.
        best = chart.position(to) - origin;
    }
    return best;
}

double GeodesicDistanceLocal(const IntrinsicTriangulation &tri, const SurfacePoint &a, const SurfacePoint &b, int maxRings)
{
    const auto log = LocalLog(tri, a, b, maxRings);
    return log ? log->norm() : std::numeric_limits<double>::infinity();
}
}  // namespace springls
