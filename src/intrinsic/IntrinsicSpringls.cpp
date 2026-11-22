#include <intrinsic/IntrinsicSpringls.hpp>

#include <intrinsic/IntrinsicSignField.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>

#ifdef SPRINGLS_OPENMP
#include <omp.h>
#endif

namespace springls
{
namespace
{
bool InParallel()
{
#ifdef SPRINGLS_OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}
// Chart epochs: unique across simulations, renewed when an instance drops stale charts.
std::atomic<uint64_t> gChartEpoch{1};

// Charts reach about three edge lengths: every local search (neighbors 0.5, fill gaps 0.5,
// contraction 1.2, arms <= 1, in h_t) stays inside one.
constexpr int kChartDepth = 6;
// A face is refined while its longest edge exceeds h_t by more than this factor.
constexpr double kRefineTolerance = 1.01;
// Largest distance from a point of a triangle to its nearest corner, over the longest edge (1/sqrt(3),
// the equilateral triangle's circumradius, rounded up): the slack of a face's distance lower bound.
constexpr double kCornerReach = 0.6;
constexpr double kWeightEpsilon = 1e-12;

double Cross(const Vec2d &a, const Vec2d &b) { return a.x() * b.y() - a.y() * b.x(); }

// Distance from x to segment [a, b], with the closest point's parameter.
double SegmentDistance(const Vec2d &a, const Vec2d &b, const Vec2d &x, double *param = nullptr)
{
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    const double t = (len2 > 0.0) ? std::clamp((x - a).dot(ab) / len2, 0.0, 1.0) : 0.0;
    if (param) *param = t;
    return (x - (a + t * ab)).norm();
}

Vec2d ComplexMultiply(double zr, double zi, const Vec2d &u) { return Vec2d(zr * u.x() - zi * u.y(), zr * u.y() + zi * u.x()); }
}  // namespace

IntrinsicSpringls::IntrinsicSpringls(const IntrinsicTriangulation &tri) : mTri(tri), mH(tri.meanEdgeLength())
{
    invalidateCharts();
    const IntrinsicTriangulation &input = mTri.input();
    for (uint32_t e = 0; e < input.edgeCount(); e++) mInputLongest = std::max(mInputLongest, input.length(e));
}

void IntrinsicSpringls::setInitialRegion(const std::vector<double> &phi) { mInitialPhi = phi; }

void IntrinsicSpringls::setVelocity(Velocity velocity, Schedule schedule)
{
    mVelocity = std::move(velocity);
    mSchedule = std::move(schedule);
}

const LocalChart &IntrinsicSpringls::chart(uint32_t face) const
{
    auto it = mCharts.find(face);
    if (it != mCharts.end()) return it->second;
    if (!InParallel()) return mCharts.emplace(face, LocalChart(mTri, face, kChartDepth)).first->second;
    // In a parallel region the shared cache is read only: a chart missing there goes to this thread's cache.
    thread_local struct {
        uint64_t epoch = 0;
        std::unordered_map<uint32_t, LocalChart> charts;
    } local;
    if (local.epoch != mChartEpoch) {
        local.charts.clear();
        local.epoch = mChartEpoch;
    }
    auto jt = local.charts.find(face);
    if (jt == local.charts.end()) jt = local.charts.emplace(face, LocalChart(mTri, face, kChartDepth)).first;
    return jt->second;
}

void IntrinsicSpringls::ensureCharts(std::vector<uint32_t> faces)
{
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    faces.erase(std::remove_if(faces.begin(), faces.end(), [&](uint32_t f) { return mCharts.count(f) != 0; }), faces.end());
    std::vector<LocalChart> built(faces.size());
    const int n = static_cast<int>(faces.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (n >= 8))
    for (int i = 0; i < n; i++) built[i].build(mTri, faces[i], kChartDepth);
    for (int i = 0; i < n; i++) mCharts.emplace(faces[i], std::move(built[i]));
}

void IntrinsicSpringls::ensureSeedCharts(std::vector<uint32_t> faces, int depth)
{
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    faces.erase(std::remove_if(faces.begin(), faces.end(),
                               [&](uint32_t f) {
                                   const auto it = mSeedCharts.find(f);
                                   return it != mSeedCharts.end() && it->second.first == depth;
                               }),
                faces.end());
    std::vector<LocalChart> built(faces.size());
    const int n = static_cast<int>(faces.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 4) if (n >= 8))
    for (int i = 0; i < n; i++) built[i].build(mTri, faces[i], depth);
    for (int i = 0; i < n; i++) mSeedCharts.insert_or_assign(faces[i], std::make_pair(depth, std::move(built[i])));
}

void IntrinsicSpringls::invalidateCharts() { mChartEpoch = gChartEpoch.fetch_add(1); }

bool IntrinsicSpringls::relative(const SurfacePoint &origin, const SurfacePoint &target, Vec2d &out) const
{
    if (origin.face == target.face) {
        out = mTri.toLayout(target) - mTri.toLayout(origin);
        return true;
    }
    const LocalChart &c = chart(origin.face);
    if (!c.contains(target.face)) return false;
    out = c.position(target) - c.position(origin);
    return true;
}

Vec2d IntrinsicSpringls::velocityAt(const SurfacePoint &p, double t) const
{
    if (!mVelocity) return Vec2d::Zero();
    const double weight = mSchedule.weight ? mSchedule.weight(t) : 1.0;
    if (std::abs(weight) < kWeightEpsilon) return Vec2d::Zero();
    return weight * mVelocity(p, t);
}

bool IntrinsicSpringls::springlInChart(const LocalChart &c, size_t s, Vec2d &a0, Vec2d &p, Vec2d &a1) const
{
    const SurfacePoint &particle = mSpringls[s].particle;
    if (!c.contains(particle.face)) return false;
    p = c.position(particle);
    a0 = p + c.vector(particle.face, mArms[s][0]);
    a1 = p + c.vector(particle.face, mArms[s][1]);
    return true;
}

const std::unordered_map<uint32_t, std::vector<uint32_t>> &IntrinsicSpringls::byFace()
{
    mByFace.clear();
    for (uint32_t s = 0; s < mSpringls.size(); s++) mByFace[mSpringls[s].particle.face].push_back(s);
    return mByFace;
}

// ---------------------------------------------------------------------------------------------
// Springls
// ---------------------------------------------------------------------------------------------

size_t IntrinsicSpringls::appendSpringl(const SurfacePoint &particle, const SurfacePoint &v0, const SurfacePoint &v1)
{
    IntrinsicSpringl s;
    s.particle = particle;
    s.v0 = v0;
    s.v1 = v1;
    Vec2d a0, a1;
    if (!relative(particle, v0, a0)) a0 = Vec2d::Zero();
    if (!relative(particle, v1, a1)) a1 = Vec2d::Zero();
    mSpringls.push_back(s);
    mArms.push_back({a0, a1});
    return mSpringls.size() - 1;
}

void IntrinsicSpringls::setArms(size_t s, const Vec2d &arm0, const Vec2d &arm1)
{
    mArms[s] = {arm0, arm1};
    mSpringls[s].v0 = Trace(mTri, mSpringls[s].particle, arm0).end;
    mSpringls[s].v1 = Trace(mTri, mSpringls[s].particle, arm1).end;
}

void IntrinsicSpringls::compact(const std::vector<uint8_t> &keep)
{
    size_t out = 0;
    for (size_t s = 0; s < mSpringls.size(); s++) {
        if (s < keep.size() && !keep[s]) continue;
        mSpringls[out] = mSpringls[s];
        mArms[out] = mArms[s];
        out++;
    }
    mSpringls.resize(out);
    mArms.resize(out);
    mNearest.clear();
}

std::vector<std::pair<SurfacePoint, SurfacePoint>> IntrinsicSpringls::springlSegments() const
{
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;
    segments.reserve(2 * mSpringls.size());
    for (const IntrinsicSpringl &s : mSpringls) {
        segments.emplace_back(s.particle, s.v0);
        segments.emplace_back(s.particle, s.v1);
    }
    return segments;
}

void IntrinsicSpringls::extractContour()
{
    mContour = ExtractIntrinsicContour(mTri, [&](uint32_t v, int) { return mPhi[v]; }, {kRegionLabel});
}

std::vector<EdgeCrossing> IntrinsicSpringls::armCrossings() const
{
    // Per springl, in parallel; concatenated in springl order.
    const int n = static_cast<int>(mSpringls.size());
    std::vector<std::vector<EdgeCrossing>> per(mSpringls.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 64) if (n >= kParallelMinHeavyItems))
    for (int s = 0; s < n; s++) {
        for (int e = 0; e < 2; e++) mTri.trace(mSpringls[s].particle, mArms[s][e], &per[s]);
    }
    std::vector<EdgeCrossing> out;
    for (const auto &list : per) out.insert(out.end(), list.begin(), list.end());
    return out;
}

void IntrinsicSpringls::refineContour()
{
    // Where a springl arm crosses the edge of a crossing, the crossing is placed on it (the arm intersection
    // nearest phi's linear zero crossing). Where none does but the bridge between two neighboring springls
    // (an arm vertex and its nearest neighbor's) does, on the bridge: the springls with their neighbor links
    // are the curve. Elsewhere the 1D slide toward the particles.
    std::unordered_map<uint32_t, std::vector<double>> onArms, onBridges;  // edge -> crossings, on its canonical halfedge
    const auto add = [&](std::unordered_map<uint32_t, std::vector<double>> &map, const EdgeCrossing &c) {
        const uint32_t edge = mTri.edge(c.halfedge);
        map[edge].push_back(mTri.edgeHalfedge(edge) == c.halfedge ? c.t : 1.0 - c.t);
    };
    for (const EdgeCrossing &c : armCrossings()) add(onArms, c);
    if (mNearest.size() != 2 * mSpringls.size()) updateNearestNeighbors();
    // The bridges' crossings per arm vertex, in parallel (charts of the vertexes' faces built first); merged in order.
    {
        std::vector<uint32_t> faces;
        faces.reserve(mNearest.size());
        for (size_t i = 0; i < mNearest.size(); i++) faces.push_back(((i % 2 == 0) ? mSpringls[i / 2].v0 : mSpringls[i / 2].v1).face);
        ensureCharts(std::move(faces));
    }
    const int nv = static_cast<int>(mNearest.size());
    std::vector<std::vector<EdgeCrossing>> bridges(mNearest.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 64) if (nv >= kParallelMinHeavyItems))
    for (int i = 0; i < nv; i++) {
        const int j = mNearest[i];
        if (j < 0) continue;
        const SurfacePoint &from = (i % 2 == 0) ? mSpringls[i / 2].v0 : mSpringls[i / 2].v1;
        const SurfacePoint &to = (j % 2 == 0) ? mSpringls[j / 2].v0 : mSpringls[j / 2].v1;
        Vec2d bridge;
        if (!relative(from, to, bridge)) continue;
        mTri.trace(from, bridge, &bridges[i]);
    }
    for (const auto &list : bridges) {
        for (const EdgeCrossing &c : list) add(onBridges, c);
    }
    std::vector<std::vector<double>> linear(mContour.size());
    for (size_t c = 0; c < mContour.size(); c++) linear[c] = mContour[c].t;
    std::vector<SurfacePoint> particles;
    particles.reserve(mSpringls.size());
    for (const IntrinsicSpringl &s : mSpringls) particles.push_back(s.particle);
    const std::vector<int> labels(particles.size(), kRegionLabel);
    RefineIntrinsicContour(mTri, mContour, particles, labels, tuning.refineIterations, tuning.refineProximity * mH, tuning.refineStepSize);
    for (size_t c = 0; c < mContour.size(); c++) {
        IntrinsicCurve &curve = mContour[c];
        for (size_t k = 0; k < curve.points.size(); k++) {
            const std::vector<double> *hits = nullptr;
            if (const auto arm = onArms.find(curve.edges[k]); arm != onArms.end()) {
                hits = &arm->second;
            } else if (const auto bridge = onBridges.find(curve.edges[k]); bridge != onBridges.end()) {
                hits = &bridge->second;
            }
            if (!hits) continue;
            double best = hits->front();
            for (double t : *hits) {
                if (std::abs(t - linear[c][k]) < std::abs(best - linear[c][k])) best = t;
            }
            curve.t[k] = std::clamp(best, 0.0, 1.0);
            curve.points[k] = IntrinsicTriangulation::halfedgePoint(mTri.edgeHalfedge(curve.edges[k]), curve.t[k]);
        }
    }
}

void IntrinsicSpringls::seedSpringls()
{
    for (const IntrinsicCurve &c : mContour) {
        for (size_t k = 0; k < c.faces.size(); k++) {
            const uint32_t f = c.faces[k];
            const SurfacePoint a = ContourPointInFace(mTri, c, k, f), b = ContourPointInFace(mTri, c, (k + 1) % c.points.size(), f);
            // A contour segment shorter than an arm at rest (the contour passing next to a mesh vertex) has no
            // reliable direction: seeded, a few of them bunch at a sharp corner and relaxation turns one across
            // the chain, which then signs the vertices beside it wrongly. Its neighbors' arms bridge it instead.
            // A closed contour whose segments are all that short (a region smaller than about an arm) gets
            // no springls at all.
            if ((mTri.toLayout(b) - mTri.toLayout(a)).norm() < tuning.armRestLength * mH) continue;
            appendSpringl(SurfacePoint{f, 0.5 * (a.bary + b.bary)}, a, b);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Adaptive refinement
// ---------------------------------------------------------------------------------------------

std::vector<uint8_t> IntrinsicSpringls::refinementMarks(const std::vector<double> &distance, double band) const
{
    std::vector<uint8_t> marks(mTri.faceCount(), 0);
    const double target = kRefineTolerance * mH;
    for (uint32_t f = 0; f < mTri.faceCount(); f++) {
        if (mTri.faceLevel(f) >= mMaxLevel) continue;
        double longest = 0.0, nearest = std::numeric_limits<double>::infinity();
        for (int c = 0; c < 3; c++) {
            longest = std::max(longest, mTri.halfedgeLength(3 * f + c));
            nearest = std::min(nearest, distance[mTri.tail(3 * f + c)]);
        }
        // Every point of the face is within kCornerReach * longest of a corner: a lower bound of its distance.
        if (longest > target && nearest - kCornerReach * longest <= band) marks[f] = 1;
    }
    return marks;
}

std::vector<SurfacePoint *> IntrinsicSpringls::heldPoints()
{
    std::vector<SurfacePoint *> points = mTrackedPoints ? mTrackedPoints() : std::vector<SurfacePoint *>{};
    points.reserve(points.size() + 3 * mSpringls.size());
    for (IntrinsicSpringl &s : mSpringls) {
        points.push_back(&s.particle);
        points.push_back(&s.v0);
        points.push_back(&s.v1);
    }
    return points;
}

void IntrinsicSpringls::reexpressArms(size_t s)
{
    // Where the particle's chart does not reach an arm vertex, the 3D difference carried into the layout
    // through the correspondence (never an arm in a stale layout).
    const SurfacePoint &p = mSpringls[s].particle;
    const SurfacePoint *ends[2] = {&mSpringls[s].v0, &mSpringls[s].v1};
    for (int e = 0; e < 2; e++) {
        Vec2d arm;
        if (!relative(p, *ends[e], arm)) {
            const IntrinsicTriangulation::ExtrinsicFrame frame = mTri.extrinsicFrame(p);
            arm = mTri.extrinsicToLayout(frame, mTri.toExtrinsic(*ends[e]).position - frame.point.position);
        }
        mArms[s][e] = arm;
    }
}

size_t IntrinsicSpringls::refineFaces(const std::vector<uint8_t> &marks)
{
    const size_t before = mTri.vertexCount();
    const std::vector<IntrinsicSpringl> old = mSpringls;
    const std::vector<SurfacePoint *> points = heldPoints();
    // Faces by corners and edges, to tell which ones the refinement changes.
    const size_t faces = mTri.faceCount();
    std::vector<uint32_t> shape(6 * faces);
    for (uint32_t he = 0; he < 3 * faces; he++) {
        shape[2 * he] = mTri.tail(he);
        shape[2 * he + 1] = mTri.edge(he);
    }
    if (mTri.refine(marks, points) == 0 && mTri.vertexCount() == before) return before;
    // Drop the charts that reach a changed face (the others unfold the same faces as before).
    std::vector<uint8_t> changed(mTri.faceCount(), 1);
    for (uint32_t f = 0; f < faces; f++) {
        bool same = true;
        for (uint32_t he = 3 * f; he < 3 * f + 3; he++) same = same && shape[2 * he] == mTri.tail(he) && shape[2 * he + 1] == mTri.edge(he);
        changed[f] = same ? 0 : 1;
    }
    invalidateCharts();  // per-thread caches may hold charts of changed faces
    for (auto it = mCharts.begin(); it != mCharts.end();) {
        const auto &chartFaces = it->second.faces();
        const bool stale = std::any_of(chartFaces.begin(), chartFaces.end(), [&](uint32_t f) { return changed[f] != 0; });
        it = stale ? mCharts.erase(it) : std::next(it);
    }
    for (auto it = mSeedCharts.begin(); it != mSeedCharts.end();) {
        const auto &chartFaces = it->second.second.faces();
        const bool stale = std::any_of(chartFaces.begin(), chartFaces.end(), [&](uint32_t f) { return changed[f] != 0; });
        it = stale ? mSeedCharts.erase(it) : std::next(it);
    }
    // The arm vertexes are relocated exactly; arms move into the particle's new face layout whenever that
    // face changed.
    for (size_t s = 0; s < mSpringls.size(); s++) {
        const SurfacePoint &p = mSpringls[s].particle;
        if (p.face == old[s].particle.face && p.bary == old[s].particle.bary && !changed[p.face]) continue;
        reexpressArms(s);
    }
    return before;
}

void IntrinsicSpringls::extendPhi(size_t from)
{
    const size_t V = mTri.vertexCount();
    std::vector<uint8_t> known(V, 1);
    for (size_t v = from; v < V; v++) known[v] = 0;
    mPhi.resize(V, 0.0);
    for (bool changed = true; changed;) {
        changed = false;
        for (uint32_t v = static_cast<uint32_t>(from); v < V; v++) {
            if (known[v]) continue;
            double sum = 0.0;
            int n = 0;
            const uint32_t first = mTri.vertexHalfedge(v);
            uint32_t he = first;
            do {
                const uint32_t w = mTri.head(he);
                if (known[w]) {
                    sum += mPhi[w];
                    n++;
                }
                he = mTri.nextOutgoing(he);
            } while (he != first);
            if (n == 0) continue;
            mPhi[v] = sum / n;
            known[v] = 1;
            changed = true;
        }
    }
}

void IntrinsicSpringls::refineInitialRegion()
{
    // Around the initial boundary: faces where phi changes sign or within the band of it. New vertices
    // take the initial phi interpolated on the input face they lie on.
    const double band = (kCapsule + kRefineMargin) * mH;
    const IntrinsicTriangulation &input = mTri.input();
    for (int pass = 0; pass < mMaxLevel; pass++) {
        std::vector<double> distance(mPhi.size());
        for (size_t v = 0; v < mPhi.size(); v++) distance[v] = std::abs(mPhi[v]);
        std::vector<uint8_t> marks = refinementMarks(distance, band);
        for (uint32_t f = 0; f < mTri.faceCount(); f++) {
            if (marks[f] || mTri.faceLevel(f) >= mMaxLevel) continue;
            bool inside = false, outside = false;
            for (int c = 0; c < 3; c++) (mPhi[mTri.tail(3 * f + c)] < 0.0 ? inside : outside) = true;
            double longest = 0.0;
            for (int c = 0; c < 3; c++) longest = std::max(longest, mTri.halfedgeLength(3 * f + c));
            if (inside && outside && longest > kRefineTolerance * mH) marks[f] = 1;
        }
        if (std::find(marks.begin(), marks.end(), 1) == marks.end()) break;
        const size_t before = refineFaces(marks);
        mPhi.resize(mTri.vertexCount());
        for (size_t v = before; v < mTri.vertexCount(); v++) {
            const IntrinsicTriangulation::ExtrinsicPoint x = mTri.vertexExtrinsic(static_cast<uint32_t>(v));
            double value = 0.0;
            for (int c = 0; c < 3; c++) value += x.bary[c] * mInitialPhi[input.tail(3 * x.face + c)];
            mPhi[v] = value;
        }
    }
}

std::vector<double> IntrinsicSpringls::refineNearSpringls()
{
    mDistance = (mCoarsening && coarseningDue()) ? rebuild() : refinementPasses();
    return mDistance;
}

bool IntrinsicSpringls::coarseningDue() const
{
    // The U kept from the last call is on the current mesh (only the refinement changes it).
    if (mDistance.size() != mTri.vertexCount()) return false;
    const double reach = coarseningDistance();
    double refined = 0.0, stale = 0.0;
    for (uint32_t f = 0; f < mTri.faceCount(); f++) {
        if (mTri.faceLevel(f) == 0) continue;
        const double area = mTri.faceArea(f);
        refined += area;
        bool beyond = true;
        for (int c = 0; c < 3 && beyond; c++) beyond = mDistance[mTri.tail(3 * f + c)] > reach;
        if (beyond) stale += area;
    }
    return stale > kCoarsenFraction * refined;
}

std::vector<double> IntrinsicSpringls::rebuild()
{
    const auto t0 = std::chrono::steady_clock::now();
    const size_t vertices = mTri.vertexCount();
    // Every held point onto the input mesh (its face and barycentric coordinates there: the same location).
    const std::vector<SurfacePoint *> points = heldPoints();
    for (SurfacePoint *p : points) {
        if (!p || p->face >= mTri.faceCount()) continue;
        const IntrinsicTriangulation::ExtrinsicPoint x = mTri.toExtrinsic(*p);
        *p = SurfacePoint{x.face, x.bary};
    }
    // The input mesh as the simulation's copy started: intrinsic Delaunay (a no-op when it already is).
    mTri.resetToInput();
    mTri.flipToDelaunay(points);
    // Input vertices keep their indices and their phi; the passes extend it to the vertices they insert.
    mPhi.resize(mTri.vertexCount());
    mCharts.clear();
    mSeedCharts.clear();
    invalidateCharts();
    mByFace.clear();
    std::vector<double> distance = refinementPasses();
    for (size_t s = 0; s < mSpringls.size(); s++) reexpressArms(s);
    mRebuilds++;
    if (verbose) {
        std::cerr << "Intrinsic Springls: rebuilt the refined mesh at t = " << mTime << ", " << vertices << " -> " << mTri.vertexCount() << " vertices, "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
    }
    return distance;
}

void IntrinsicSpringls::rebuildMesh()
{
    mDistance = rebuild();
    // The contour lived on the old mesh: back from the level set carried over, onto the springls.
    extractContour();
    refineContour();
}

std::vector<double> IntrinsicSpringls::refinementPasses()
{
    // U, the distance to the springls (fast marching from exact chart seeds), marks the faces to refine;
    // the U of the final mesh is the reconstruction's.
    const double band = (kCapsule + kRefineMargin) * mH;
    for (int pass = 0;; pass++) {
        SpringlSegments segments;
        segments.segments = springlSegments();
        // The march stops beyond the band plus the corner reach of the longest edge: every face the
        // refinement criterion can mark, and every value the clamped phi uses, is still reached.
        double longest = 0.0;
        for (uint32_t e = 0; e < mTri.edgeCount(); e++) longest = std::max(longest, mTri.length(e));
        SignFieldParams params{kCapsule * mH, kBandWidth * mH};
        params.march = std::max(params.clamp, band) + kCornerReach * longest;
        // With coarsening, far enough to tell the stale faces (U above the coarsening distance).
        if (mCoarsening) params.march = std::max(params.march, coarseningDistance());
        {
            // The seed charts (one per particle face: every segment starts at its particle), built in parallel.
            std::vector<uint32_t> faces(mSpringls.size());
            for (size_t i = 0; i < mSpringls.size(); i++) faces[i] = mSpringls[i].particle.face;
            ensureSeedCharts(std::move(faces), SeedChartDepth(mTri, params));
        }
        std::vector<double> distance = SpringlDistance(mTri, segments, params, [this](uint32_t face, int depth) -> const LocalChart & {
            auto it = mSeedCharts.find(face);
            if (it == mSeedCharts.end() || it->second.first != depth)
                it = mSeedCharts.insert_or_assign(face, std::make_pair(depth, LocalChart(mTri, face, depth))).first;
            return it->second.second;
        });
        if (mSpringls.empty() || pass > mMaxLevel) return distance;
        const std::vector<uint8_t> marks = refinementMarks(distance, band);
        if (std::find(marks.begin(), marks.end(), 1) == marks.end()) return distance;
        extendPhi(refineFaces(marks));
    }
}

int IntrinsicSpringls::springlSide(uint32_t v, double reach) const
{
    // v's side (the region on the springls' left) of the nearest springl in v's chart. When the nearest point
    // is a springl's outer end, the pseudo-normal there: that arm's normal plus the normal of the arm its
    // nearest neighbor (the link relaxation pulls together) ends with, as at a vertex of a polyline; 0 at a
    // free end (no neighbor), where the side is ambiguous.
    const SurfacePoint corner = mTri.vertexPoint(v);
    const LocalChart &c = chart(corner.face);
    const Vec2d x = c.position(corner);
    const bool linked = mNearest.size() == 2 * mSpringls.size();
    const auto normal = [](const Vec2d &from, const Vec2d &to) {
        const Vec2d d = to - from;
        const double length = d.norm();
        return (length > 0.0) ? Vec2d(Vec2d(-d.y(), d.x()) / length) : Vec2d(Vec2d::Zero());
    };
    double best = reach;
    int side = 0;
    for (uint32_t g : c.faces()) {
        const auto it = mByFace.find(g);
        if (it == mByFace.end()) continue;
        for (uint32_t s : it->second) {
            Vec2d a0, p, a1;
            if (!springlInChart(c, s, a0, p, a1)) continue;
            double t0 = 0.0, t1 = 0.0;
            const double d0 = SegmentDistance(a0, p, x, &t0), d1 = SegmentDistance(p, a1, x, &t1);
            const bool first = d0 <= d1;
            const double d = first ? d0 : d1;
            if (!(d < best)) continue;
            best = d;
            const Vec2d foot = first ? Vec2d(a0 + t0 * (p - a0)) : Vec2d(p + t1 * (a1 - p));
            Vec2d n = first ? normal(a0, p) : normal(p, a1);
            if (first && t1 <= 0.0 && t0 >= 1.0) n += normal(p, a1);  // at the particle: both arms
            const int end = (first && t0 <= 0.0) ? 0 : (!first && t1 >= 1.0) ? 1 : -1;
            if (end >= 0) {
                const int j = linked ? mNearest[2 * s + end] : -1;
                Vec2d b0, q, b1;
                if (j < 0 || !springlInChart(c, static_cast<size_t>(j) / 2, b0, q, b1)) {
                    side = 0;
                    continue;
                }
                n += (j % 2 == 0) ? normal(b0, q) : normal(q, b1);
            }
            const double dot = n.dot(x - foot);
            side = (dot > 0.0) ? -1 : (dot < 0.0) ? 1 : 0;
        }
    }
    return side;
}

void IntrinsicSpringls::pruneCharts(const std::vector<double> &distance)
{
    // Every chart lookup is centered on a face within the band of the springls (particles, arm vertexes,
    // contour, walls): charts of faces the springls have left are dropped, so the caches follow the band
    // instead of growing with the swept area.
    const double band = (kCapsule + kRefineMargin) * mH;
    const auto near = [&](uint32_t f) {
        if (f >= mTri.faceCount()) return false;
        for (int c = 0; c < 3; c++) {
            if (distance[mTri.tail(3 * f + c)] <= band) return true;
        }
        return false;
    };
    for (auto it = mCharts.begin(); it != mCharts.end();) it = near(it->first) ? std::next(it) : mCharts.erase(it);
    for (auto it = mSeedCharts.begin(); it != mSeedCharts.end();) it = near(it->first) ? std::next(it) : mSeedCharts.erase(it);
}

void IntrinsicSpringls::reconstructPhi(const std::vector<double> &distance)
{
    SignFieldParams params{kCapsule * mH, kBandWidth * mH};
    params.correction = kSideCorrection * mH;
    byFace();
    // The side test runs in parallel from the walls' charts: built here first (in parallel).
    {
        std::vector<uint32_t> faces;
        for (uint32_t v = 0; v < mTri.vertexCount(); v++) {
            if (distance[v] <= params.capsule) faces.push_back(mTri.vertexPoint(v).face);
        }
        ensureCharts(std::move(faces));
    }
    SignField field = SignFieldFromDistance(mTri, distance, params, [this](uint32_t v, double reach) { return springlSide(v, reach); });
    if (field.leaked) {
        // The springls no longer enclose a region: keep the previous level set for this sub-step.
        mLeaks++;
        if (verbose) std::cerr << "Intrinsic Springls: sign field leaked at t = " << mTime << ", keeping the previous level set" << std::endl;
        return;
    }
    mPhi = std::move(field.phi);
}

bool IntrinsicSpringls::init(std::string &error)
{
    if (mInitialPhi.empty() || mInitialPhi.size() != mTri.vertexCount()) {
        error = "the initial region needs one value per vertex of the triangulation";
        return false;
    }
    if (std::none_of(mInitialPhi.begin(), mInitialPhi.end(), [](double v) { return v < 0.0; })) {
        error = "the initial region is empty (no vertex has phi < 0)";
        return false;
    }
    const double base = (mLengthScale > 0.0) ? mLengthScale : mTri.meanEdgeLength();
    mH = base / std::ldexp(1.0, mMaxLevel);
    mTime = 0.0;
    mIteration = 0;
    mContractedTotal = mFilledTotal = mLeaks = 0;
    mRebuilds = 0;
    mDistance.clear();  // the initial refinement changes the mesh
    mEvents.clear();
    mSpringls.clear();
    mArms.clear();
    mNearest.clear();
    mCharts.clear();
    mSeedCharts.clear();
    invalidateCharts();
    mPhi = mInitialPhi;
    refineInitialRegion();
    // The crossings interpolate the initial distances, so the initial contour needs no refinement.
    extractContour();
    seedSpringls();
    relaxAll();
    updateNearestNeighbors();
    return true;
}

bool IntrinsicSpringls::step()
{
    int contracted = 0, filledTotal = 0;
    for (int iter = 0; iter < tuning.advectionSubstepsPerStep; iter++) {
        if (mSchedule.weight) {
            const double remaining = mSchedule.endTime - mTime;
            if (remaining <= 0.0) break;
            const double dt = advect(tuning.maxStepPixels, std::min(remaining, mSchedule.maxTimeStep));
            // Land exactly on endTime instead of accumulating rounding.
            mTime = (dt >= remaining) ? mSchedule.endTime : mTime + dt;
        } else {
            mTime += advect(tuning.maxStepPixels, std::numeric_limits<double>::infinity());
        }
        // Semi-Lagrangian rebuild: refine where the springls are, then the level set from the springls.
        const std::vector<double> distance = refineNearSpringls();
        reconstructPhi(distance);
        pruneCharts(distance);
        extractContour();
        refineContour();
        relaxWithRememberedNeighbors();
        contracted += contract();
        updateNearestNeighbors();
        constexpr int kMaxFillPasses = 16;
        int filled = 0, passes = 0;
        do {
            passes++;
            filled = fill();
            relaxAll();
            filledTotal += filled;
        } while (filled > 0 && passes < kMaxFillPasses);
        // The relaxation after the last fill pass moved the arms: the contour back onto the springls.
        extractContour();
        refineContour();
    }
    mContractedTotal += static_cast<size_t>(contracted);
    mFilledTotal += static_cast<size_t>(filledTotal);
    mIteration++;
    if (verbose && (mIteration % SpringLevelSet2D::kLogInterval) == 0) {
        std::cerr << "Intrinsic Springls: step " << mIteration << " t = " << mTime << ", springls " << mSpringls.size() << ", contracted " << contracted
                  << ", filled " << filledTotal << ", vertices " << mTri.vertexCount() << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}

// ---------------------------------------------------------------------------------------------
// Advection and relaxation
// ---------------------------------------------------------------------------------------------

double IntrinsicSpringls::advect(double maxStep, double maxTimeStep)
{
    const size_t N = mSpringls.size();
    const double t0 = mTime;
    std::vector<Vec2d> k1(N);
    // The particles' fastest speed with the schedule's weight left out.
    double maxSpeed = 0.0;
    const int n = static_cast<int>(N);
    SPRINGLS_OMP(omp parallel for reduction(max : maxSpeed) if (n >= kParallelMinHeavyItems))
    for (int i = 0; i < n; i++) {
        k1[i] = mVelocity ? mVelocity(mSpringls[i].particle, t0) : Vec2d::Zero();
        maxSpeed = std::max(maxSpeed, k1[i].norm());
    }
    const auto weightAt = [&](double t) {
        if (!mSchedule.weight) return 1.0;
        const double w = mSchedule.weight(t);
        return (std::abs(w) < kWeightEpsilon) ? 0.0 : w;
    };
    const double w0 = weightAt(t0);
    for (Vec2d &v : k1) v *= w0;
    // As SpringLevelSet2D::advectionTimeStep, in units of h̄: the fastest particle moves maxStep h̄. The
    // speed is bounded over the whole step by the larger |weight| at its two ends (exact for a weight
    // monotone in |w| on each side of a reversal, such as cos(πt/T) over steps shorter than T/2), so a
    // step that starts as the flow slows to its reversal cannot jump past it at the speed of the start.
    const auto stepFor = [&](double absWeight) {
        const double speed = absWeight * maxSpeed / mH;
        return std::min((speed > 1.0) ? maxStep / speed : maxStep, maxTimeStep);
    };
    double dt = stepFor(std::abs(w0));
    for (int pass = 0; pass < 8; pass++) {
        const double shorter = stepFor(std::max(std::abs(w0), std::abs(weightAt(t0 + dt))));
        if (!(shorter < dt)) break;
        dt = shorter;
    }
    const double tHalf = t0 + 0.5 * dt, tEnd = t0 + dt;
    // RK4 on the surface: stage points by the exponential map, stage velocities transported back.
    const auto displace = [&](const SurfacePoint &x, const Vec2d &v1) {
        const TraceResult r2 = Trace(mTri, x, (0.5 * dt) * v1);
        const Vec2d v2 = TransportBack(r2, velocityAt(r2.end, tHalf));
        const TraceResult r3 = Trace(mTri, x, (0.5 * dt) * v2);
        const Vec2d v3 = TransportBack(r3, velocityAt(r3.end, tHalf));
        const TraceResult r4 = Trace(mTri, x, dt * v3);
        const Vec2d v4 = TransportBack(r4, velocityAt(r4.end, tEnd));
        return Trace(mTri, x, (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4));
    };
    // Each springl's particle and arm vertexes move on their own (in parallel); then the charts of the
    // particles' new faces (in parallel); then each springl's fit (in parallel).
    std::vector<std::array<TraceResult, 3>> moved(N);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 16) if (n >= kParallelMinHeavyItems))
    for (int i = 0; i < n; i++) {
        const IntrinsicSpringl &s = mSpringls[i];
        moved[i] = {displace(s.particle, k1[i]), displace(s.v0, velocityAt(s.v0, t0)), displace(s.v1, velocityAt(s.v1, t0))};
    }
    {
        std::vector<uint32_t> faces(N);
        for (size_t i = 0; i < N; i++) faces[i] = moved[i][0].end.face;
        ensureCharts(std::move(faces));
    }
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 16) if (n >= kParallelMinHeavyItems))
    for (int i = 0; i < n; i++) {
        IntrinsicSpringl &s = mSpringls[i];
        const TraceResult &rp = moved[i][0];
        // The springl becomes the similarity transform (rotation + uniform scale about its advected
        // particle) of its previous shape that best fits its advected vertexes.
        s.particle = rp.end;
        const Vec2d arms[2] = {Rotate(mArms[i][0], rp.rotation), Rotate(mArms[i][1], rp.rotation)};
        const SurfacePoint ends[2] = {moved[i][1].end, moved[i][2].end};
        double re = 0.0, im = 0.0, norm = 0.0;
        for (int e = 0; e < 2; e++) {
            Vec2d w;
            if (!relative(s.particle, ends[e], w)) w = arms[e];
            re += arms[e].dot(w);
            im += Cross(arms[e], w);
            norm += arms[e].squaredNorm();
        }
        constexpr double kMinFit = 1e-24;
        if (norm < kMinFit * mH * mH || re * re + im * im < kMinFit * norm * norm) {
            setArms(i, arms[0], arms[1]);
            continue;
        }
        const double zr = re / norm, zi = im / norm;
        setArms(i, ComplexMultiply(zr, zi, arms[0]), ComplexMultiply(zr, zi, arms[1]));
    }
    updateNearestNeighbors();
    return dt;
}

void IntrinsicSpringls::updateNearestNeighbors()
{
    const double maxDistance = tuning.nearestNeighborDistance * mH;
    const size_t V = 2 * mSpringls.size();
    const auto vertex = [&](size_t i) -> const SurfacePoint & { return (i % 2 == 0) ? mSpringls[i / 2].v0 : mSpringls[i / 2].v1; };
    std::unordered_map<uint32_t, std::vector<uint32_t>> vertexesByFace;
    for (uint32_t i = 0; i < V; i++) vertexesByFace[vertex(i).face].push_back(i);
    mNearest.assign(V, -1);
    {
        std::vector<uint32_t> faces(V);
        for (size_t i = 0; i < V; i++) faces[i] = vertex(i).face;
        ensureCharts(std::move(faces));
    }
    const int nv = static_cast<int>(V);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 32) if (nv >= kParallelMinHeavyItems))
    for (int i = 0; i < nv; i++) {
        const SurfacePoint &q = vertex(i);
        const LocalChart &c = chart(q.face);
        const Vec2d x = c.position(q);
        double best = maxDistance;
        int bestIndex = -1;
        for (uint32_t f : c.faces()) {
            const auto it = vertexesByFace.find(f);
            if (it == vertexesByFace.end()) continue;
            for (uint32_t j : it->second) {
                if (j / 2 == static_cast<uint32_t>(i) / 2) continue;
                const double d = (c.position(vertex(j)) - x).norm();
                if (d < best || (d == best && bestIndex >= 0 && static_cast<int>(j) < bestIndex)) {
                    best = d;
                    bestIndex = static_cast<int>(j);
                }
            }
        }
        mNearest[i] = bestIndex;
    }
}

void IntrinsicSpringls::relax(double timeStep)
{
    // As SpringLevelSet2D::relaxSpringl, in each particle's face layout, lengths in h̄.
    constexpr double kMaxForce = 0.999;
    const double kMinLength = 0.001 * mH;
    const double armRestLength = tuning.armRestLength * mH, maxArmLength = tuning.maxArmLength * mH;
    const double restSeparation = tuning.neighborRestSeparation * mH, extent = SpringLevelSet2D::kExtent * mH;
    const size_t N = mSpringls.size();
    std::vector<std::array<Vec2d, 2>> updated(N);
    {
        std::vector<uint32_t> faces(N);
        for (size_t idx = 0; idx < N; idx++) faces[idx] = mSpringls[idx].particle.face;
        ensureCharts(std::move(faces));
    }
    const int n = static_cast<int>(N);
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 32) if (n >= kParallelMinHeavyItems))
    for (int idx = 0; idx < n; idx++) {
        const IntrinsicSpringl &s = mSpringls[idx];
        Vec2d tangents[2], motion[2];
        double armLength[2], springForce[2], moment = 0.0;
        for (int e = 0; e < 2; e++) {
            const Vec2d arm = mArms[idx][e];
            armLength[e] = arm.norm();
            tangents[e] = (armLength[e] > 1e-12 * mH) ? Vec2d(arm / armLength[e]) : arm;
            Vec2d velocity = Vec2d::Zero();
            const int nbr = mNearest[2 * idx + e];
            if (nbr >= 0) {
                const IntrinsicSpringl &o = mSpringls[static_cast<size_t>(nbr) / 2];
                Vec2d q;
                if (relative(s.particle, (nbr % 2 == 0) ? o.v0 : o.v1, q)) {
                    const Vec2d dir = q - arm;
                    const double w = std::atanh(kMaxForce * std::clamp((dir.norm() - restSeparation) / (extent + restSeparation), -1.0, 1.0));
                    velocity += w * dir;
                }
            }
            motion[e] = timeStep * tuning.sharpness * velocity;
            springForce[e] = timeStep * tuning.springConstant * (armRestLength - armLength[e]);
            // 2D measures the moment in pixels as an angle; here in units of h̄.
            moment += Cross(motion[e], tangents[e]) / mH;
        }
        const double cosa = std::cos(moment), sina = std::sin(moment);
        for (int e = 0; e < 2; e++) {
            const double radial = std::clamp(armLength[e] + motion[e].dot(tangents[e]) + springForce[e], kMinLength, maxArmLength);
            const Vec2d p = radial * tangents[e];
            updated[idx][e] = Vec2d(p.x() * cosa + p.y() * sina, -p.x() * sina + p.y() * cosa);
        }
    }
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 32) if (n >= kParallelMinHeavyItems))
    for (int idx = 0; idx < n; idx++) setArms(idx, updated[idx][0], updated[idx][1]);
}

void IntrinsicSpringls::relaxWithRememberedNeighbors()
{
    if (mNearest.size() != 2 * mSpringls.size()) updateNearestNeighbors();
    for (int i = 0; i < tuning.relaxIterations; i++) relax(tuning.relaxTimeStep);
}

void IntrinsicSpringls::relaxAll()
{
    updateNearestNeighbors();
    relaxWithRememberedNeighbors();
}

// ---------------------------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------------------------

int IntrinsicSpringls::contract()
{
    // Contour segments by face, for distances from each springl to the contour.
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> segmentsByFace;
    for (uint32_t c = 0; c < mContour.size(); c++) {
        for (uint32_t k = 0; k < mContour[c].faces.size(); k++) segmentsByFace[mContour[c].faces[k]].emplace_back(c, k);
    }
    const double contractDistance = tuning.contractDistance * mH, maxArm = tuning.maxArmLength * mH;
    const double minEdge = SpringLevelSet2D::kMinEdgeLength * mH;
    std::vector<uint8_t> keep(mSpringls.size(), 1);
    int contracted = 0;
    // Each springl's distance to the contour, in parallel; the contractions in order.
    const int n = static_cast<int>(mSpringls.size());
    std::vector<double> contourDistance(mSpringls.size());
    {
        std::vector<uint32_t> faces(mSpringls.size());
        for (size_t s = 0; s < mSpringls.size(); s++) faces[s] = mSpringls[s].particle.face;
        ensureCharts(std::move(faces));
    }
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 32) if (n >= kParallelMinHeavyItems))
    for (int s = 0; s < n; s++) {
        const IntrinsicSpringl &sp = mSpringls[s];
        const LocalChart &c = chart(sp.particle.face);
        const Vec2d x = c.position(sp.particle);
        double d = std::numeric_limits<double>::infinity();
        for (uint32_t f : c.faces()) {
            const auto it = segmentsByFace.find(f);
            if (it == segmentsByFace.end()) continue;
            for (const auto &[ci, k] : it->second) {
                const IntrinsicCurve &curve = mContour[ci];
                const Vec2d a = c.position(ContourPointInFace(mTri, curve, k, f));
                const Vec2d b = c.position(ContourPointInFace(mTri, curve, (k + 1) % curve.points.size(), f));
                d = std::min(d, SegmentDistance(a, b, x));
            }
        }
        contourDistance[s] = d;
    }
    for (size_t s = 0; s < mSpringls.size(); s++) {
        const IntrinsicSpringl &sp = mSpringls[s];
        const double d = contourDistance[s];
        const double d1 = mArms[s][0].norm(), d2 = mArms[s][1].norm();
        const bool ok = d <= contractDistance && d1 >= minEdge && d2 >= minEdge && d1 <= maxArm && d2 <= maxArm;
        if (!ok) {
            keep[s] = 0;
            contracted++;
            ResampleEvent event;
            event.kind = (d > contractDistance)         ? ResampleEvent::Kind::ContractFar
                         : (std::min(d1, d2) < minEdge) ? ResampleEvent::Kind::ContractShortArm
                                                        : ResampleEvent::Kind::ContractLongArm;
            event.time = mTime;
            event.position = mTri.toExtrinsic(sp.particle).position;
            event.distance = d / mH;
            event.separation = std::max(d1, d2) / mH;
            mEvents.push_back(event);
        }
    }
    if (contracted > 0) compact(keep);
    return contracted;
}

double IntrinsicSpringls::springlHole(const LocalChart &c, const Vec2d &x, double reach, size_t existing) const
{
    // The arm vertexes within `reach` of x: the largest distance from one of them to the nearest arm vertex of
    // another springl (a chain end at a hole has none near; inside a chain the neighbor is close).
    std::vector<std::pair<Vec2d, uint32_t>> vertexes;
    for (uint32_t g : c.faces()) {
        const auto it = mByFace.find(g);
        if (it == mByFace.end()) continue;
        for (uint32_t s : it->second) {
            Vec2d a0, p, a1;
            if (s >= existing || !springlInChart(c, s, a0, p, a1)) continue;
            vertexes.emplace_back(a0, s);
            vertexes.emplace_back(a1, s);
        }
    }
    double hole = 0.0;
    for (const auto &[y, s] : vertexes) {
        if ((y - x).norm() > reach) continue;
        double nearest = std::numeric_limits<double>::infinity();
        for (const auto &[z, o] : vertexes) {
            if (o != s) nearest = std::min(nearest, (z - y).norm());
        }
        hole = std::max(hole, nearest);
    }
    return hole;
}

int IntrinsicSpringls::fill()
{
    extractContour();
    refineContour();
    // Gap tests see the springls that exist before this pass.
    const auto buckets = byFace();
    const size_t existing = mSpringls.size();
    const double fillDistance = tuning.fillDistance * mH;
    int filled = 0;
    // Every contour segment's gap to the existing springls, in parallel (each test sees only the springls that
    // existed before this pass); then the fills in contour order.
    std::vector<std::pair<uint32_t, uint32_t>> segments;  // (curve, index)
    for (uint32_t c = 0; c < mContour.size(); c++) {
        for (uint32_t k = 0; k < mContour[c].faces.size(); k++) segments.emplace_back(c, k);
    }
    {
        std::vector<uint32_t> faces(segments.size());
        for (size_t i = 0; i < segments.size(); i++) faces[i] = mContour[segments[i].first].faces[segments[i].second];
        ensureCharts(std::move(faces));
    }
    const int n = static_cast<int>(segments.size());
    std::vector<double> gaps(segments.size());
    SPRINGLS_OMP(omp parallel for schedule(dynamic, 32) if (n >= kParallelMinHeavyItems))
    for (int i = 0; i < n; i++) {
        const IntrinsicCurve &curve = mContour[segments[i].first];
        const size_t k = segments[i].second;
        const uint32_t f = curve.faces[k];
        const SurfacePoint a = ContourPointInFace(mTri, curve, k, f), b = ContourPointInFace(mTri, curve, (k + 1) % curve.points.size(), f);
        const SurfacePoint m{f, 0.5 * (a.bary + b.bary)};
        const LocalChart &c = chart(f);
        const Vec2d x = c.position(m);
        double gap = std::numeric_limits<double>::infinity();
        for (uint32_t g : c.faces()) {
            const auto it = buckets.find(g);
            if (it == buckets.end()) continue;
            for (uint32_t s : it->second) {
                if (s >= existing) continue;
                Vec2d a0, p, a1;
                springlInChart(c, s, a0, p, a1);
                gap = std::min(gap, SegmentDistance(a0, a1, x));
            }
        }
        gaps[i] = gap;
    }
    for (int i = 0; i < n; i++) {
        const double gap = gaps[i];
        if (!(gap > fillDistance)) continue;
        const IntrinsicCurve &curve = mContour[segments[i].first];
        const size_t k = segments[i].second;
        const uint32_t f = curve.faces[k];
        const SurfacePoint a = ContourPointInFace(mTri, curve, k, f), b = ContourPointInFace(mTri, curve, (k + 1) % curve.points.size(), f);
        const SurfacePoint m{f, 0.5 * (a.bary + b.bary)};
        const LocalChart &c = chart(f);
        const Vec2d x = c.position(m);
        // As seeding: no springl on a segment shorter than an arm at rest (no reliable direction).
        if ((c.position(b) - c.position(a)).norm() < tuning.armRestLength * mH) continue;
        ResampleEvent event;
        event.kind = ResampleEvent::Kind::Fill;
        event.time = mTime;
        event.position = mTri.toExtrinsic(m).position;
        event.distance = gap / mH;
        event.separation = springlHole(c, x, gap + fillDistance, existing) / mH;
        event.length = (c.position(b) - c.position(a)).norm() / mH;
        mEvents.push_back(event);
        appendSpringl(m, a, b);
        filled++;
    }
    if (filled > 0) mNearest.clear();
    return filled;
}
}  // namespace springls
