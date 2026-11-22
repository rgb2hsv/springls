#include <intrinsic/IntrinsicLoopScenario.hpp>

#include <intrinsic/IntrinsicDistance.hpp>
#include <intrinsic/IntrinsicTrace.hpp>
#include <mesh/HodgeDecomposition.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace springls
{
namespace
{
constexpr double kPiD = 3.141592653589793;
constexpr double kTwoPiD = 2.0 * kPiD;
// Outline samples per mean edge length.
constexpr double kSamplesPerEdge = 4.0;

Vec2d Perp(const Vec2d &a) { return Vec2d(-a.y(), a.x()); }
double Cross(const Vec2d &a, const Vec2d &b) { return a.x() * b.y() - a.y() * b.x(); }

// Whether segments pq and ab intersect (touching counts).
bool SegmentsCross(const Vec2d &p, const Vec2d &q, const Vec2d &a, const Vec2d &b)
{
    const auto cross = [](const Vec2d &u, const Vec2d &w) { return u.x() * w.y() - u.y() * w.x(); };
    const double d1 = cross(q - p, a - p), d2 = cross(q - p, b - p), d3 = cross(b - a, p - a), d4 = cross(b - a, q - a);
    return ((d1 >= 0.0) != (d2 >= 0.0) || d1 == 0.0 || d2 == 0.0) && ((d3 >= 0.0) != (d4 >= 0.0) || d3 == 0.0 || d4 == 0.0);
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

// Appends points along the arc of circle (center, radius) from angle a0 to a1 (signed sweep),
// excluding the start point.
void AppendArc(std::vector<Vec2d> &out, const Vec2d &center, double radius, double a0, double a1, double spacing)
{
    const int n = std::max(2, static_cast<int>(std::ceil(std::abs(a1 - a0) * radius / spacing)));
    for (int k = 1; k <= n; k++) {
        const double a = a0 + (a1 - a0) * k / n;
        out.push_back(center + radius * Vec2d(std::cos(a), std::sin(a)));
    }
}

void AppendLine(std::vector<Vec2d> &out, const Vec2d &to, double spacing)
{
    const Vec2d from = out.back();
    const int n = std::max(1, static_cast<int>(std::ceil((to - from).norm() / spacing)));
    for (int k = 1; k <= n; k++) out.push_back(from + (to - from) * (static_cast<double>(k) / n));
}

// A polyline mapped onto the surface and split into segments inside single faces; the edges it crosses.
struct MappedCurve {
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;
    std::unordered_set<uint32_t> crossedEdges;
    std::vector<uint32_t> exits;  // per crossing, in order: the halfedge of the face the curve leaves
};

bool MapPolyline(const IntrinsicTriangulation &tri, const SurfacePoint &center, const Vec2d &X, const Vec2d &Y, const std::vector<Vec2d> &points, bool closed,
                 MappedCurve &curve, std::string &error)
{
    std::vector<SurfacePoint> mapped;
    for (const Vec2d &p : points) mapped.push_back(Trace(tri, center, p.x() * X + p.y() * Y).end);
    const size_t n = closed ? mapped.size() : mapped.size() - 1;
    for (size_t k = 0; k < n; k++) {
        const SurfacePoint &a = mapped[k], &b = mapped[(k + 1) % mapped.size()];
        const auto log = LocalLog(tri, a, b, 3);
        if (!log) {
            error = "outline samples are too far apart on the surface";
            return false;
        }
        std::vector<EdgeCrossing> crossings;
        const TraceResult r = Trace(tri, a, *log, &crossings);
        SurfacePoint from = a;
        for (const EdgeCrossing &c : crossings) {
            curve.segments.emplace_back(from, IntrinsicTriangulation::halfedgePoint(c.halfedge, c.t));
            curve.crossedEdges.insert(tri.edge(c.halfedge));
            curve.exits.push_back(c.halfedge);
            from = IntrinsicTriangulation::halfedgePoint(tri.twin(c.halfedge), 1.0 - c.t);
        }
        curve.segments.emplace_back(from, r.end);
    }
    return true;
}
}  // namespace

std::vector<Vec2d> PuzzlePieceOutline(double r, int samplesPerSide)
{
    const double spacing = 2.0 * r / std::max(samplesPerSide, 4);
    const double R = 0.35 * r, offset = 0.25 * r;
    const double half = std::sqrt(R * R - offset * offset);  // where a knob's circle meets its side
    const auto angle = [](const Vec2d &from, const Vec2d &to) { return std::atan2(to.y() - from.y(), to.x() - from.x()); };
    // Each knob arc runs the long way round its circle, between the two points where it meets the side.
    const auto knob = [&](std::vector<Vec2d> &out, const Vec2d &center, const Vec2d &enter, const Vec2d &leave, bool counterClockwise) {
        AppendLine(out, enter, spacing);
        const double a0 = angle(center, enter);
        double a1 = angle(center, leave);
        if (counterClockwise && a1 <= a0) a1 += kTwoPiD;
        if (!counterClockwise && a1 >= a0) a1 -= kTwoPiD;
        AppendArc(out, center, R, a0, a1, spacing);
        out.back() = leave;
    };
    std::vector<Vec2d> out = {Vec2d(-r, -r)};
    knob(out, Vec2d(0, -r + offset), Vec2d(-half, -r), Vec2d(half, -r), false);  // bottom socket, cut in
    AppendLine(out, Vec2d(r, -r), spacing);
    knob(out, Vec2d(r + offset, 0), Vec2d(r, -half), Vec2d(r, half), true);  // front tab
    AppendLine(out, Vec2d(r, r), spacing);
    knob(out, Vec2d(0, r + offset), Vec2d(half, r), Vec2d(-half, r), true);  // top tab
    AppendLine(out, Vec2d(-r, r), spacing);
    knob(out, Vec2d(-r + offset, 0), Vec2d(-r, half), Vec2d(-r, -half), false);  // back socket, cut in
    AppendLine(out, Vec2d(-r, -r), spacing);
    out.pop_back();  // the start point
    return out;
}

std::vector<Vec2d> StarOutline(double r, int samples)
{
    std::vector<Vec2d> out;
    for (int k = 0; k < samples; k++) {
        const double theta = kTwoPiD * k / samples;
        out.push_back(r * (1.0 + 0.35 * std::cos(5.0 * theta)) * Vec2d(std::cos(theta), std::sin(theta)));
    }
    return out;
}

bool ObjectFromOutline(const IntrinsicTriangulation &tri, const SurfacePoint &center, const Vec2d &xAxis, const std::vector<Vec2d> &outline,
                       std::vector<double> &phi, std::string &error)
{
    if (outline.size() < 3 || xAxis.norm() == 0.0) {
        error = "outline needs at least three points and a direction";
        return false;
    }
    if (!InsidePolygon(outline, Vec2d::Zero())) {
        error = "outline must contain its origin";
        return false;
    }
    const Vec2d X = xAxis.normalized(), Y = Perp(X);
    MappedCurve curve;
    if (!MapPolyline(tri, center, X, Y, outline, true, curve, error)) return false;
    const std::vector<double> distance = FastMarch(tri, SeedsFromSegments(tri, curve.segments));
    // Flood fill from the center's face corner nearest the center that no outline segment separates from it.
    uint32_t start = UINT32_MAX;
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int a, int b) { return center.bary[a] > center.bary[b]; });
    const auto P = tri.layout(center.face);
    const Vec2d x = tri.toLayout(center);
    for (int c : order) {
        bool separated = false;
        for (const auto &[a, b] : curve.segments) {
            if (a.face != center.face) continue;
            if (SegmentsCross(x, P[c], tri.toLayout(a), tri.toLayout(b))) separated = true;
        }
        if (!separated) {
            start = tri.tail(3 * center.face + c);
            break;
        }
    }
    if (start == UINT32_MAX) {
        error = "no vertex inside the object (object too small for the mesh)";
        return false;
    }
    std::vector<uint8_t> inside(tri.vertexCount(), 0);
    std::deque<uint32_t> queue = {start};
    inside[start] = 1;
    size_t filled = 1;
    while (!queue.empty()) {
        const uint32_t v = queue.front();
        queue.pop_front();
        const uint32_t first = tri.vertexHalfedge(v);
        uint32_t he = first;
        do {
            const uint32_t w = tri.head(he);
            if (!inside[w] && !curve.crossedEdges.count(tri.edge(he))) {
                inside[w] = 1;
                queue.push_back(w);
                filled++;
            }
            he = tri.nextOutgoing(he);
        } while (he != first);
    }
    // Whether the flood stayed inside the outline: not by the region's size (the object may be the larger
    // side of the surface) but by the outline's orientation. The inside is on the left of a
    // counter-clockwise outline (the chart X, Y = Perp(X) and the face layouts keep the orientation). Leaving
    // a (counter-clockwise) face through halfedge he, head(he) is on the curve's left and tail(he) on its
    // right, so on every edge the outline crosses once, exactly one endpoint is inside: the one the
    // orientation says. A leaked flood (both sides inside) or an outline that overlaps itself on the
    // surface (wrapping past the cut locus) breaks this on some edges; a simple closed outline never does.
    double signedArea = 0.0;
    for (size_t k = 0; k < outline.size(); k++) signedArea += Cross(outline[k], outline[(k + 1) % outline.size()]);
    const bool counterClockwise = signedArea > 0.0;
    std::unordered_map<uint32_t, int> crossingsPerEdge;
    for (uint32_t he : curve.exits) crossingsPerEdge[tri.edge(he)]++;
    size_t agree = 0, edges = 0;
    for (uint32_t he : curve.exits) {
        if (crossingsPerEdge[tri.edge(he)] != 1) continue;
        const bool left = inside[tri.head(he)] != 0, right = inside[tri.tail(he)] != 0;
        agree += (left == counterClockwise) && (right != counterClockwise);
        edges++;
    }
    if (filled == tri.vertexCount() || edges == 0 || agree != edges) {
        error = "the object's outline does not enclose a region on the mesh (it overlaps itself, or the flood fill leaked)";
        return false;
    }
    phi.resize(tri.vertexCount());
    for (size_t v = 0; v < phi.size(); v++) phi[v] = inside[v] ? -distance[v] : distance[v];
    return true;
}

bool LoopScenario::setup(const IntrinsicTriangulation &tri, const LoopConfig &config, std::string &error)
{
    mTri = &tri;
    mConfig = config;
    if (!(config.speed > 0.0)) {
        error = "speed must be positive";
        return false;
    }
    const Points3 &P = tri.inputVertices();
    uint32_t top = 0, bottom = 0;
    Vec3d lo = P[0].cast<double>(), hi = lo;
    for (uint32_t v = 1; v < P.size(); v++) {
        if (P[v].z() > P[top].z()) top = v;
        if (P[v].z() < P[bottom].z()) bottom = v;
        lo = lo.cwiseMin(P[v].cast<double>());
        hi = hi.cwiseMax(P[v].cast<double>());
    }
    mAxisCenter = 0.5 * (lo + hi);
    mCenter = top;
    mD = FastMarch(tri, {{mCenter, 0.0}});
    mRho = (config.loopDistance > 0.0) ? config.loopDistance : config.loopFraction * 0.5 * mD[bottom];
    mRadius = (config.radius > 0.0) ? config.radius : mRho / 6.0;
    // Object center: distance ρ from c along the geodesic leaving c toward +x.
    const uint32_t first = tri.vertexHalfedge(mCenter);
    uint32_t he = first, best = first;
    double bestScore = -1e30;
    Vec2d direction(1, 0);
    const Vec3d pc = P[mCenter].cast<double>();
    do {
        const Vec3d a = P[tri.head(he)].cast<double>() - pc, b = P[tri.tail(tri.prev(he))].cast<double>() - pc;
        const Vec3d n = a.cross(b).normalized();
        const Vec3d target = (Vec3d(1, 0, 0) - n.dot(Vec3d(1, 0, 0)) * n);
        Eigen::Matrix<double, 3, 2> E;
        E << a, b;
        const Eigen::Vector2d w = (E.transpose() * E).ldlt().solve(E.transpose() * target);
        // Inside the corner's wedge scores highest; otherwise how close.
        const double score = std::min(w[0], w[1]);
        if (score > bestScore) {
            bestScore = score;
            best = he;
            const auto L = tri.layout(tri.face(he));
            const int c = static_cast<int>(he % 3);
            direction = w[0] * (L[(c + 1) % 3] - L[c]) + w[1] * (L[(c + 2) % 3] - L[c]);
        }
        he = tri.nextOutgoing(he);
    } while (he != first);
    const SurfacePoint start = IntrinsicTriangulation::cornerPoint(tri.face(best), static_cast<int>(best % 3));
    mObjectCenter = Trace(tri, start, mRho * direction.normalized()).end;
    // The rotation: the object's center moves at `speed`.
    const Vec3d x = tri.toExtrinsic(mObjectCenter).position - mAxisCenter;
    const double axisDistance = std::hypot(x.x(), x.y());
    if (!(axisDistance > 1e-9 * (hi - lo).norm())) {
        error = "the object's center lies on the rotation axis: move it with --loop-distance";
        return false;
    }
    mOmega = config.speed / axisDistance;
    mPeriod = kPiD * kTwoPiD / mOmega;  // π × one revolution's time at full speed (schedule())
    mFaceVelocity.clear();
    mVertexVelocity.clear();
    mStream.clear();
    mExactFraction = mRemainderFraction = 0.0;
    if (config.flow == LoopFlow::Hodge && !setupHodge(tri, error)) return false;
    const Vec2d xAxis = velocity(mObjectCenter, 0.0);
    if (xAxis.norm() == 0.0) {
        error = "the loop direction vanishes at the object's center";
        return false;
    }
    const double h = (config.lengthScale > 0.0) ? config.lengthScale : tri.meanEdgeLength();
    const int samples = std::max(64, static_cast<int>(std::ceil(kSamplesPerEdge * 2.0 * kTwoPiD * mRadius / h)));
    const std::vector<Vec2d> outline =
        (config.object == LoopObject::Star) ? StarOutline(mRadius, samples) : PuzzlePieceOutline(mRadius, std::max(16, samples / 8));
    if (!ObjectFromOutline(tri, mObjectCenter, xAxis, outline, mPhi, error)) return false;
    mInitialArea = RegionArea(tri, [&](uint32_t v) { return mPhi[v]; });
    return true;
}

bool LoopScenario::setupHodge(const IntrinsicTriangulation &tri, std::string &error)
{
    // The input mesh's triangles (input face f is the intrinsic input triangulation's face f).
    const IntrinsicTriangulation &input = tri.input();
    const Points3 &P = tri.inputVertices();
    Triangles triangles(input.faceCount());
    std::vector<Vec3d> rotation(input.faceCount());
    for (uint32_t f = 0; f < input.faceCount(); f++) {
        triangles[f] = {input.tail(3 * f), input.tail(3 * f + 1), input.tail(3 * f + 2)};
        const Vec3d centroid = (P[triangles[f][0]] + P[triangles[f][1]] + P[triangles[f][2]]).cast<double>() / 3.0;
        rotation[f] = mOmega * Vec3d::UnitZ().cross(centroid - mAxisCenter);  // DecomposeHodge takes the tangential part
    }
    HodgeDecomposition hodge;
    if (!DecomposeHodge(P, triangles, rotation, hodge, error)) {
        error = "Hodge flow: " + error;
        return false;
    }
    const double norm = FaceFieldNorm(P, triangles, rotation);
    mExactFraction = FaceFieldNorm(P, triangles, hodge.exact) / norm;
    mRemainderFraction = FaceFieldNorm(P, triangles, hodge.remainder) / norm;
    // Scale so the object's center moves at `speed`.
    const uint32_t centerFace = tri.extrinsicFrame(mObjectCenter).point.face;
    const double centerSpeed = hodge.coexact[centerFace].norm();
    if (!(centerSpeed > 1e-12 * norm)) {
        error = "the Hodge flow vanishes at the object's center: move it with --loop-distance";
        return false;
    }
    const double scale = mConfig.speed / centerSpeed;
    mFaceVelocity = std::move(hodge.coexact);
    for (Vec3d &v : mFaceVelocity) v *= scale;
    mStream = std::move(hodge.psi);
    for (double &s : mStream) s *= scale;
    if (mConfig.smoothVelocity) {
        mVertexVelocity.assign(P.size(), Vec3d::Zero());
        std::vector<double> weight(P.size(), 0.0);
        for (uint32_t f = 0; f < input.faceCount(); f++) {
            const Vec3d a = P[triangles[f][0]].cast<double>(), b = P[triangles[f][1]].cast<double>(), c = P[triangles[f][2]].cast<double>();
            const double area = 0.5 * (b - a).cross(c - a).norm();
            for (int q = 0; q < 3; q++) {
                mVertexVelocity[triangles[f][q]] += area * mFaceVelocity[f];
                weight[triangles[f][q]] += area;
            }
        }
        for (size_t v = 0; v < P.size(); v++) {
            if (weight[v] > 0.0) mVertexVelocity[v] /= weight[v];
        }
        // The smoothed field's own speed at the object's center sets the scale (all the fields alike).
        const double smoothSpeed = velocity(tri, mObjectCenter, 0.0).norm();
        if (!(smoothSpeed > 1e-12 * mConfig.speed)) {
            error = "the smoothed Hodge flow vanishes at the object's center: move it with --loop-distance";
            return false;
        }
        const double rescale = mConfig.speed / smoothSpeed;
        for (Vec3d &v : mVertexVelocity) v *= rescale;
        for (Vec3d &v : mFaceVelocity) v *= rescale;
        for (double &s : mStream) s *= rescale;
    }

    // One revolution: RK2 along the center's streamline until it comes back (the closest approach to the
    // start after having left it, from a parabola through the last three squared distances).
    const double h = tri.meanEdgeLength();
    const double dt = 0.25 * h / mConfig.speed;
    const Vec3d start = tri.toExtrinsic(mObjectCenter).position;
    SurfacePoint x = mObjectCenter;
    double d2[3] = {0.0, 0.0, 0.0};
    bool left = false;
    const int maxSteps = 4000000;
    for (int k = 1; k <= maxSteps; k++) {
        const Vec2d v1 = velocity(tri, x, 0.0);
        const TraceResult half = Trace(tri, x, (0.5 * dt) * v1);
        const Vec2d v2 = TransportBack(half, velocity(tri, half.end, 0.0));
        x = Trace(tri, x, dt * v2).end;
        d2[0] = d2[1];
        d2[1] = d2[2];
        d2[2] = (tri.toExtrinsic(x).position - start).squaredNorm();
        // Left: farther than 4 h; back: a local minimum within 2 h (the RK2 drift over a loop stays well under it).
        if (d2[2] > 16.0 * h * h) left = true;
        if (left && k >= 3 && d2[1] < 4.0 * h * h && d2[2] > d2[1] && d2[1] <= d2[0]) {
            // Minimum of the parabola through (k-2, d2[0]), (k-1, d2[1]), (k, d2[2]).
            const double curvature = d2[0] - 2.0 * d2[1] + d2[2];
            const double offset = curvature > 0.0 ? std::clamp(0.5 * (d2[0] - d2[2]) / curvature, -1.0, 1.0) : 0.0;
            mPeriod = kPiD * (static_cast<double>(k - 1) + offset) * dt;
            return true;
        }
    }
    error = "the Hodge flow's streamline through the object's center does not close";
    return false;
}

Vec2d LoopScenario::velocity(const IntrinsicTriangulation &tri, const SurfacePoint &p, double) const
{
    const IntrinsicTriangulation::ExtrinsicFrame frame = tri.extrinsicFrame(p);
    if (!mVertexVelocity.empty()) {
        const IntrinsicTriangulation &input = tri.input();
        Vec3d v = Vec3d::Zero();
        for (int q = 0; q < 3; q++) v += frame.point.bary[q] * mVertexVelocity[input.tail(3 * frame.point.face + q)];
        return tri.extrinsicToLayout(frame, v);
    }
    if (!mFaceVelocity.empty()) return tri.extrinsicToLayout(frame, mFaceVelocity[frame.point.face]);
    return tri.extrinsicToLayout(frame, mOmega * Vec3d::UnitZ().cross(frame.point.position - mAxisCenter));
}

IntrinsicSpringls::Schedule LoopScenario::schedule() const
{
    IntrinsicSpringls::Schedule s;
    const double period = mPeriod;
    s.weight = [period](double t) { return static_cast<float>(std::cos(kPiD * t / period)); };
    s.endTime = mPeriod;
    return s;
}

LoopScenario::Metrics LoopScenario::measure(const IntrinsicSpringls &sim) const
{
    const IntrinsicTriangulation &tri = sim.triangulation();
    const IntrinsicTriangulation &input = tri.input();
    Metrics m;
    m.time = sim.time();
    m.springls = sim.springls().size();
    const double area = ContourArea(tri, sim.contour(), IntrinsicSpringls::kRegionLabel, [&](uint32_t v, int) { return sim.phi()[v]; });
    m.areas.push_back(area);
    m.relativeAreaErrors.push_back(area / mInitialArea - 1.0);
    const double h = sim.meanEdgeLength();
    double sum = 0.0;
    size_t n = 0;
    for (const IntrinsicCurve &c : sim.contour()) {
        for (const SurfacePoint &p : c.points) {
            // The initial signed distance interpolated on the input mesh face the point lies on, then its
            // magnitude (zero on the initial contour; |phi| interpolated would not be, across a sign change).
            const IntrinsicTriangulation::ExtrinsicPoint x = tri.toExtrinsic(p);
            double phi = 0.0;
            for (int q = 0; q < 3; q++) phi += x.bary[q] * mPhi[input.tail(3 * x.face + q)];
            const double d = std::abs(phi) / h;
            sum += d;
            m.maxShapeError = std::max(m.maxShapeError, d);
            n++;
        }
    }
    m.meanShapeError = (n > 0) ? sum / static_cast<double>(n) : 0.0;
    return m;
}
}  // namespace springls
