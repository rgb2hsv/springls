// IntrinsicSpringls: one region of springls on an adaptively refined intrinsic triangulation, its level
// set rebuilt from the springls every sub-step (semi-Lagrangian).
#include "Check.hpp"
#include "IntrinsicTestRegions.hpp"

#include <intrinsic/IntrinsicContour.hpp>
#include <intrinsic/IntrinsicReference.hpp>
#include <intrinsic/IntrinsicSpringls.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <unordered_map>

using namespace springls;

namespace
{
const Vec3d kStarCenter = Vec3d(1, 0.137, 0.0713);

// The region's area bounded by the simulation's (refined) contour.
double Area(const IntrinsicSpringls &sim)
{
    return ContourArea(sim.triangulation(), sim.contour(), IntrinsicSpringls::kRegionLabel, [&](uint32_t v, int) { return sim.phi()[v]; });
}

std::vector<std::pair<SurfacePoint, SurfacePoint>> ContourSegments(const IntrinsicTriangulation &tri, const std::vector<IntrinsicCurve> &curves)
{
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;
    for (const IntrinsicCurve &c : curves) {
        for (size_t k = 0; k < c.faces.size(); k++) {
            segments.emplace_back(ContourPointInFace(tri, c, k, c.faces[k]), ContourPointInFace(tri, c, (k + 1) % c.points.size(), c.faces[k]));
        }
    }
    return segments;
}

std::vector<std::pair<SurfacePoint, SurfacePoint>> SpringlSegmentsOf(const IntrinsicSpringls &sim)
{
    std::vector<std::pair<SurfacePoint, SurfacePoint>> segments;
    for (const IntrinsicSpringl &s : sim.springls()) {
        segments.emplace_back(s.v0, s.particle);
        segments.emplace_back(s.particle, s.v1);
    }
    return segments;
}

double LongestEdge(const IntrinsicTriangulation &tri)
{
    double longest = 0.0;
    for (uint32_t e = 0; e < tri.edgeCount(); e++) longest = std::max(longest, tri.length(e));
    return longest;
}

// Rotation about z (a Killing field of the sphere) at a point of the simulation's triangulation.
IntrinsicSpringls::Velocity Rotation(const IntrinsicSpringls &sim)
{
    return [&sim](const SurfacePoint &p, double) {
        const IntrinsicTriangulation &tri = sim.triangulation();
        return springls_test::TangentToLayout(tri, p.face, Vec3d(0, 0, 1).cross(springls_test::CornerPosition(tri, p)));
    };
}
// The star rotating about z (the mesh refines along its path), init done.
void StartRotatingStar(IntrinsicSpringls &sim, int maxLevel, bool coarsening)
{
    sim.setMaxLevel(maxLevel);
    sim.setCoarsening(coarsening);
    sim.setInitialRegion(springls_test::StarPhi(sim.triangulation(), kStarCenter, 0.3));
    IntrinsicSpringls::Schedule schedule;
    schedule.weight = [](double) { return 1.0f; };
    sim.setVelocity(Rotation(sim), schedule);
    std::string error;
    CHECK(sim.init(error));
}

// 3D positions of every particle, arm vertex and given point.
std::vector<Vec3d> HeldPositions(const IntrinsicSpringls &sim, const std::vector<SurfacePoint> &points)
{
    const IntrinsicTriangulation &tri = sim.triangulation();
    std::vector<Vec3d> out;
    for (const IntrinsicSpringl &s : sim.springls()) {
        for (const SurfacePoint *p : {&s.particle, &s.v0, &s.v1}) out.push_back(tri.toExtrinsic(*p).position);
    }
    for (const SurfacePoint &p : points) out.push_back(tri.toExtrinsic(p).position);
    return out;
}

// Largest 3D distance from each arm's trace from its particle to its arm vertex.
double WorstArm(const IntrinsicSpringls &sim)
{
    const IntrinsicTriangulation &tri = sim.triangulation();
    double worst = 0.0;
    for (size_t s = 0; s < sim.springls().size(); s++) {
        const IntrinsicSpringl &sp = sim.springls()[s];
        const SurfacePoint *ends[2] = {&sp.v0, &sp.v1};
        for (int e = 0; e < 2; e++) {
            const SurfacePoint end = Trace(tri, sp.particle, sim.arms()[s][e]).end;
            worst = std::max(worst, (tri.toExtrinsic(end).position - tri.toExtrinsic(*ends[e]).position).norm());
        }
    }
    return worst;
}

std::vector<SurfacePoint> RandomPoints(const IntrinsicTriangulation &tri, size_t n)
{
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> face(0, static_cast<uint32_t>(tri.faceCount() - 1));
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<SurfacePoint> out;
    for (size_t k = 0; k < n; k++) {
        double a = unit(rng), b = unit(rng);
        if (a + b > 1.0) {
            a = 1.0 - a;
            b = 1.0 - b;
        }
        out.push_back(SurfacePoint{face(rng), Vec3d(a, b, 1.0 - a - b)});
    }
    return out;
}
}  // namespace

TEST(springls_single_region_seed)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    std::string error;
    CHECK(sim.init(error));
    const IntrinsicTriangulation &refined = sim.triangulation();
    const double h = sim.meanEdgeLength();
    CHECK_NEAR(h, 0.5 * tri.meanEdgeLength(), 1e-15);
    CHECK(refined.vertexCount() > tri.vertexCount());
    CHECK(sim.contour().size() == 1 && sim.contour()[0].closed && sim.contour()[0].label == IntrinsicSpringls::kRegionLabel);
    const auto contour = ContourSegments(refined, sim.contour());
    // One springl per contour segment at least an arm's rest length long, at its midpoint (relaxation moves
    // only the arms); shorter segments get none.
    std::vector<Vec3d> midpoints;
    for (const auto &[a, b] : contour) {
        if ((refined.toLayout(b) - refined.toLayout(a)).norm() < sim.tuning.armRestLength * h) continue;
        midpoints.push_back(refined.toExtrinsic(SurfacePoint{a.face, 0.5 * (a.bary + b.bary)}).position);
    }
    CHECK(sim.springls().size() == midpoints.size());
    CHECK(sim.springls().size() > 100 && midpoints.size() < contour.size());
    std::vector<uint8_t> used(midpoints.size(), 0);
    double worst = 0.0;
    for (const IntrinsicSpringl &s : sim.springls()) {
        const Vec3d x = refined.toExtrinsic(s.particle).position;
        size_t best = 0;
        for (size_t k = 1; k < midpoints.size(); k++) {
            if ((midpoints[k] - x).norm() < (midpoints[best] - x).norm()) best = k;
        }
        worst = std::max(worst, (midpoints[best] - x).norm());
        used[best] = 1;
    }
    std::printf("  seed: %zu springls, particle to segment midpoint %.2g h\n", sim.springls().size(), worst / h);
    CHECK(worst < 1e-9 * h);
    CHECK(std::count(used.begin(), used.end(), 1) == static_cast<long>(used.size()));
    CHECK(sim.phi().size() == refined.vertexCount());
}

TEST(springls_refine_near_object)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(2);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    std::string error;
    CHECK(sim.init(error));
    const IntrinsicTriangulation &refined = sim.triangulation();
    const double h = sim.meanEdgeLength();
    CHECK_NEAR(h, 0.25 * tri.meanEdgeLength(), 1e-15);
    const double longestInput = LongestEdge(tri);
    const std::vector<double> U = FastMarch(refined, SeedsFromSegments(refined, SpringlSegmentsOf(sim)));
    // Near the springls the edges reach the target length: h_t, or the longest input edge split maxLevel times
    // where the input edges are longer than the mean (the level cap stops there).
    const double target = 1.01 * std::max(h, longestInput / 4.0);
    size_t near = 0;
    double longestNear = 0.0, sumNear = 0.0;
    for (uint32_t e = 0; e < refined.edgeCount(); e++) {
        const uint32_t he = refined.edgeHalfedge(e);
        if (U[refined.tail(he)] > 2.0 * h || U[refined.head(he)] > 2.0 * h) continue;
        near++;
        longestNear = std::max(longestNear, refined.length(e));
        sumNear += refined.length(e);
    }
    std::printf("  refine: %zu -> %zu vertices, %zu near edges, longest %.3f h_t, mean %.3f h_t (input longest / 4 = %.3f h_t)\n", tri.vertexCount(),
                refined.vertexCount(), near, longestNear / h, sumNear / std::max<size_t>(near, 1) / h, longestInput / 4.0 / h);
    CHECK(near > 300);
    CHECK(longestNear <= target);
    CHECK(sumNear / near <= 1.01 * h);
    int deepest = 0;
    for (uint32_t f = 0; f < refined.faceCount(); f++) deepest = std::max(deepest, refined.faceLevel(f));
    CHECK(deepest == 2);
    // Far from the springls nothing changed.
    const double far = (IntrinsicSpringls::kBandWidth + 2.0) * h + 1.5 * longestInput;
    size_t farFaces = 0;
    for (uint32_t f = 0; f < refined.faceCount(); f++) {
        double d = 1e300;
        for (int c = 0; c < 3; c++) d = std::min(d, U[refined.tail(3 * f + c)]);
        if (d <= far) continue;
        farFaces++;
        CHECK(refined.faceLevel(f) == 0);
    }
    CHECK(farFaces > 1000);
}

TEST(springls_zero_velocity_no_resampling)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(2);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    sim.setVelocity([](const SurfacePoint &, double) { return Vec2d::Zero(); }, {});
    std::string error;
    CHECK(sim.init(error));
    const double before = Area(sim);
    const size_t initial = sim.springls().size();
    for (int i = 0; i < 10; i++) sim.step();
    const double after = Area(sim);
    const size_t resampled = sim.contractedTotal() + sim.filledTotal();
    std::printf("  zero velocity: area error %.5f, springls %zu -> %zu, contracted %zu, filled %zu, leaks %zu\n", after / before - 1.0, initial,
                sim.springls().size(), sim.contractedTotal(), sim.filledTotal(), sim.leakCount());
    CHECK(std::abs(after - before) < 0.005 * before);
    CHECK(resampled * 20 < initial);
    CHECK(sim.leakCount() == 0);
    CHECK(sim.iteration() == 10);
}

TEST(springls_object_larger_than_half)
{
    // The complement of a star: the object covers most of the sphere. The exterior flood starts in the
    // largest region, which is inside; the springls' orientation says so, and the region keeps its sign.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    std::vector<double> phi = springls_test::StarPhi(tri, kStarCenter, 0.3);
    for (double &v : phi) v = -v;
    sim.setInitialRegion(phi);
    sim.setVelocity([](const SurfacePoint &, double) { return Vec2d::Zero(); }, {});
    std::string error;
    CHECK(sim.init(error));
    const double before = Area(sim);
    double total = 0.0;
    for (uint32_t f = 0; f < tri.faceCount(); f++) total += tri.faceArea(f);
    const size_t initial = sim.springls().size();
    for (int i = 0; i < 10; i++) sim.step();
    const double after = Area(sim);
    std::printf("  complement: %.1f%% of the surface, area error %.5f, contracted %zu, filled %zu, leaks %zu\n", 100.0 * before / total, after / before - 1.0,
                sim.contractedTotal(), sim.filledTotal(), sim.leakCount());
    CHECK(before > 0.5 * total);
    CHECK(std::abs(after - before) < 0.005 * before);
    CHECK((sim.contractedTotal() + sim.filledTotal()) * 20 < initial);
    CHECK(sim.leakCount() == 0);
}

TEST(springls_rigid_rotation_single_region)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(2);
    // Off the equator (colatitude about 33 degrees): slower, and a shorter path to refine, than at the equator.
    sim.setInitialRegion(springls_test::StarPhi(tri, Vec3d(0.5, 0.07, 0.85), 0.3));
    IntrinsicSpringls::Schedule schedule;
    schedule.weight = [](double) { return 1.0f; };
    schedule.endTime = 2.0 * kPi;
    sim.setVelocity(Rotation(sim), schedule);
    // The initial contour, carried through every refinement of the run.
    IntrinsicReference initial;
    sim.setTrackedPoints([&] { return initial.points(); });
    std::string error;
    CHECK(sim.init(error));
    const double h = sim.meanEdgeLength();
    const double before = Area(sim);
    initial.initialize(sim.triangulation(), sim.contour(), 1);
    const auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    while (sim.step() && steps < 5000) steps++;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK_NEAR(sim.time(), 2.0 * kPi, 1e-9);
    const double after = Area(sim);
    const IntrinsicTriangulation &refined = sim.triangulation();
    // Exact (chart) distance from every contour point to the initial contour.
    std::vector<SurfacePoint> points;
    for (const IntrinsicCurve &c : sim.contour()) points.insert(points.end(), c.points.begin(), c.points.end());
    double shape = 0.0;
    for (double d : initial.distancesAt(refined, points, 20.0 * h)) shape += d;
    shape /= static_cast<double>(std::max<size_t>(points.size(), 1));
    std::printf("  rotation: %d steps, %.1f s, %zu vertices, area error %.4f, mean shape error %.3f h_t, springls %zu, contracted %zu, filled %zu, leaks %zu\n",
                steps, seconds, refined.vertexCount(), after / before - 1.0, shape / h, sim.springls().size(), sim.contractedTotal(), sim.filledTotal(),
                sim.leakCount());
    CHECK(std::abs(after - before) < 0.02 * before);
    CHECK(shape < 0.5 * h);
}

TEST(springls_keep_sign_on_leak)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    sim.setVelocity([](const SurfacePoint &, double) { return Vec2d::Zero(); }, {});
    std::string error;
    CHECK(sim.init(error));
    const double h = sim.meanEdgeLength();
    const double before = Area(sim);
    // A gap of about 6 h_t (wider than two capsules, 3 h_t) in the springls.
    const IntrinsicTriangulation &refined = sim.triangulation();
    const Vec3d x = refined.toExtrinsic(sim.springls().front().particle).position;
    std::vector<uint8_t> keep(sim.springls().size(), 1);
    size_t removed = 0;
    for (size_t s = 0; s < keep.size(); s++) {
        if ((refined.toExtrinsic(sim.springls()[s].particle).position - x).norm() < 3.0 * h) {
            keep[s] = 0;
            removed++;
        }
    }
    sim.removeSpringls(keep);
    CHECK(removed >= 8);
    sim.step();
    const double after = Area(sim);
    std::printf("  leak: removed %zu springls, leaks %zu, area error %.5f, filled %zu, springls %zu\n", removed, sim.leakCount(), after / before - 1.0,
                sim.filledTotal(), sim.springls().size());
    CHECK(sim.leakCount() >= 1);
    CHECK(std::abs(after - before) < 0.01 * before);
    CHECK(sim.filledTotal() >= removed / 2);
}

TEST(springls_neighbors_found)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    sim.setInitialRegion(springls_test::StarPhi(tri, Vec3d(0.12, 1, 0.3), 0.45));
    std::string error;
    CHECK(sim.init(error));
    const auto &nbr = sim.nearestNeighbors();
    CHECK(nbr.size() == 2 * sim.springls().size());
    size_t found = 0;
    for (size_t i = 0; i < nbr.size(); i++) {
        if (nbr[i] < 0) continue;
        found++;
        CHECK(static_cast<size_t>(nbr[i]) / 2 != i / 2);
    }
    CHECK(found > nbr.size() / 2);
}

TEST(springls_length_scale_sets_units)
{
    // Every length constant is in units of h_t: the length scale (default: the mean edge length) over
    // 2^maxLevel.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    const std::vector<double> phi = springls_test::StarPhi(tri, kStarCenter, 0.45);
    double times[2];
    for (int k = 0; k < 2; k++) {
        IntrinsicSpringls sim(tri);
        sim.setMaxLevel(0);
        sim.setLengthScale((k + 1) * 0.5 * tri.meanEdgeLength());
        sim.setInitialRegion(phi);
        sim.setVelocity(Rotation(sim), {});
        std::string error;
        CHECK(sim.init(error));
        CHECK_NEAR(sim.meanEdgeLength(), (k + 1) * 0.5 * tri.meanEdgeLength(), 1e-15);
        CHECK(sim.triangulation().vertexCount() == tri.vertexCount());
        sim.step();
        times[k] = sim.time();
    }
    // A step moves the fastest particle maxStep length scales per sub-step.
    CHECK(std::abs(times[1] / times[0] - 2.0) < 0.05);
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    sim.setLengthScale(0.8 * tri.meanEdgeLength());
    sim.setInitialRegion(phi);
    std::string error;
    CHECK(sim.init(error));
    CHECK_NEAR(sim.meanEdgeLength(), 0.4 * tri.meanEdgeLength(), 1e-15);
}

TEST(springls_arms_follow_refinement)
{
    // Through a refining init and steps that refine ahead of the moving star, every arm stays in its
    // particle's face layout: tracing the particle along it lands on the arm vertex.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(2);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    IntrinsicSpringls::Schedule schedule;
    schedule.weight = [](double) { return 1.0f; };
    sim.setVelocity(Rotation(sim), schedule);
    std::string error;
    CHECK(sim.init(error));
    const double h = sim.meanEdgeLength();
    const size_t initial = sim.triangulation().vertexCount();
    double worst = 0.0;
    const auto check = [&] {
        const IntrinsicTriangulation &refined = sim.triangulation();
        for (size_t s = 0; s < sim.springls().size(); s++) {
            const IntrinsicSpringl &sp = sim.springls()[s];
            const SurfacePoint *ends[2] = {&sp.v0, &sp.v1};
            for (int e = 0; e < 2; e++) {
                const SurfacePoint end = Trace(refined, sp.particle, sim.arms()[s][e]).end;
                worst = std::max(worst, (refined.toExtrinsic(end).position - refined.toExtrinsic(*ends[e]).position).norm());
            }
        }
    };
    check();
    for (int step = 0; step < 8; step++) {
        sim.step();
        check();
    }
    std::printf("  arms: %zu -> %zu vertices, worst %.3g h_t\n", initial, sim.triangulation().vertexCount(), worst / h);
    CHECK(sim.triangulation().vertexCount() > initial);  // the steps refined
    CHECK(worst < 1e-9 * h);
}

TEST(contour_crossings_on_springls)
{
    // A star rotated a few steps (springls off their seed contour): every contour crossing on an edge that
    // a springl arm crosses lies on that springl.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(1);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    IntrinsicSpringls::Schedule schedule;
    schedule.weight = [](double) { return 1.0f; };
    sim.setVelocity(Rotation(sim), schedule);
    std::string error;
    CHECK(sim.init(error));
    for (int i = 0; i < 5; i++) sim.step();
    const IntrinsicTriangulation &refined = sim.triangulation();
    const double h = sim.meanEdgeLength();
    // Every arm's crossings, as parameters on each edge's canonical halfedge.
    std::unordered_map<uint32_t, std::vector<double>> crossed;
    for (size_t s = 0; s < sim.springls().size(); s++) {
        for (int e = 0; e < 2; e++) {
            std::vector<EdgeCrossing> crossings;
            refined.trace(sim.springls()[s].particle, sim.arms()[s][e], &crossings);
            for (const EdgeCrossing &c : crossings) {
                const uint32_t edge = refined.edge(c.halfedge);
                crossed[edge].push_back(refined.edgeHalfedge(edge) == c.halfedge ? c.t : 1.0 - c.t);
            }
        }
    }
    size_t onSpringls = 0, total = 0;
    double worst = 0.0;
    for (const IntrinsicCurve &c : sim.contour()) {
        for (size_t k = 0; k < c.points.size(); k++) {
            total++;
            const auto it = crossed.find(c.edges[k]);
            if (it == crossed.end()) continue;
            double best = 1e300;
            for (double t : it->second) best = std::min(best, std::abs(t - c.t[k]) * refined.length(c.edges[k]));
            worst = std::max(worst, best);
            onSpringls++;
        }
    }
    std::printf("  %zu of %zu crossings on springl-crossed edges, worst distance %.3g h_t\n", onSpringls, total, worst / h);
    CHECK(onSpringls > total / 2);
    CHECK(worst <= 1e-9 * h);
}

TEST(coarsen_rebuild_matches_one_pass)
{
    // The mesh refined along the star's path (no coarsening), then rebuilt: the rebuild is the one-pass
    // refinement of the input around the current springls (a second rebuild gives the same mesh), and faces
    // beyond the coarsening distance are input faces again.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    StartRotatingStar(sim, 2, false);
    for (int i = 0; i < 30; i++) sim.step();
    const IntrinsicTriangulation &refined = sim.triangulation();
    const size_t swept = refined.vertexCount();
    const auto shape = [&] {
        std::vector<double> lengths;
        for (uint32_t e = 0; e < refined.edgeCount(); e++) lengths.push_back(refined.length(e));
        std::sort(lengths.begin(), lengths.end());
        return lengths;
    };
    const auto t0 = std::chrono::steady_clock::now();
    sim.rebuildMesh();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const size_t vertices = refined.vertexCount(), faces = refined.faceCount();
    const std::vector<double> lengths = shape();
    sim.rebuildMesh();
    CHECK(sim.rebuildCount() == 2);
    CHECK(refined.vertexCount() == vertices && refined.faceCount() == faces);
    const std::vector<double> again = shape();
    double worst = (again.size() == lengths.size()) ? 0.0 : 1e300;
    for (size_t k = 0; k < std::min(again.size(), lengths.size()); k++) worst = std::max(worst, std::abs(again[k] - lengths[k]));
    CHECK(worst <= 1e-12);
    const std::vector<double> U = FastMarch(refined, SeedsFromSegments(refined, SpringlSegmentsOf(sim)));
    size_t far = 0, farRefined = 0;
    for (uint32_t f = 0; f < refined.faceCount(); f++) {
        bool beyond = true;
        for (int c = 0; c < 3; c++) beyond = beyond && U[refined.tail(3 * f + c)] > sim.coarseningDistance();
        if (!beyond) continue;
        far++;
        farRefined += refined.faceLevel(f) > 0;
    }
    std::printf("  rebuild: %zu vertices swept -> %zu (input %zu), %.3f s, edge lengths differ by %.3g, %zu far faces, %zu refined\n", swept, vertices,
                tri.vertexCount(), seconds, worst, far, farRefined);
    CHECK(far > 1000);
    CHECK(farRefined == 0);
    CHECK(vertices < swept);
}

TEST(coarsen_relocates_points_exactly)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    std::vector<SurfacePoint> tracked;
    sim.setTrackedPoints([&] {
        std::vector<SurfacePoint *> out;
        for (SurfacePoint &p : tracked) out.push_back(&p);
        return out;
    });
    StartRotatingStar(sim, 2, false);
    tracked = RandomPoints(sim.triangulation(), 50);
    for (int i = 0; i < 30; i++) sim.step();
    const std::vector<Vec3d> before = HeldPositions(sim, tracked);
    sim.rebuildMesh();
    const std::vector<Vec3d> after = HeldPositions(sim, tracked);
    CHECK(after.size() == before.size());
    double worst = 0.0;
    for (size_t k = 0; k < std::min(before.size(), after.size()); k++) worst = std::max(worst, (after[k] - before[k]).norm());
    std::printf("  relocate: %zu points, worst move %.3g input edges\n", before.size(), worst / tri.meanEdgeLength());
    CHECK(worst < 1e-9 * tri.meanEdgeLength());
}

TEST(coarsen_keeps_arms)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    StartRotatingStar(sim, 2, false);
    for (int i = 0; i < 30; i++) sim.step();
    sim.rebuildMesh();
    const double worst = WorstArm(sim);
    std::printf("  arms after a rebuild: worst %.3g h_t\n", worst / sim.meanEdgeLength());
    CHECK(worst < 1e-9 * sim.meanEdgeLength());
}

TEST(coarsen_points_on_input_edges)
{
    // Points exactly on input edges and at input vertices, near the star (in the refined region) and far.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    std::vector<SurfacePoint> tracked;
    for (uint32_t f = 0; f < tri.faceCount(); f++) {
        const double d = (springls_test::CornerPosition(tri, IntrinsicTriangulation::cornerPoint(f, 0)) - kStarCenter.normalized()).norm();
        if (d > 0.45 && f % 97 != 0) continue;
        tracked.push_back(SurfacePoint{f, Vec3d(0.3, 0.7, 0.0)});
        tracked.push_back(IntrinsicTriangulation::cornerPoint(f, static_cast<int>(f % 3)));
    }
    sim.setTrackedPoints([&] {
        std::vector<SurfacePoint *> out;
        for (SurfacePoint &p : tracked) out.push_back(&p);
        return out;
    });
    StartRotatingStar(sim, 2, false);
    for (int i = 0; i < 10; i++) sim.step();
    const std::vector<Vec3d> before = HeldPositions(sim, tracked);
    sim.rebuildMesh();
    const std::vector<Vec3d> after = HeldPositions(sim, tracked);
    const IntrinsicTriangulation &refined = sim.triangulation();
    bool valid = true;
    for (const SurfacePoint &p : tracked) {
        valid = valid && p.face < refined.faceCount() && std::abs(p.bary.sum() - 1.0) < 1e-12 && p.bary.minCoeff() >= -1e-12;
    }
    double worst = 0.0;
    for (size_t k = 0; k < before.size(); k++) worst = std::max(worst, (after[k] - before[k]).norm());
    std::printf("  %zu points on input edges and vertices: worst move %.3g input edges\n", tracked.size(), worst / tri.meanEdgeLength());
    CHECK(tracked.size() > 100);
    CHECK(valid);
    CHECK(worst < 1e-9 * tri.meanEdgeLength());
}

TEST(coarsen_never_rebuilds_still_object)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    sim.setMaxLevel(2);
    sim.setInitialRegion(springls_test::StarPhi(tri, kStarCenter, 0.3));
    sim.setVelocity([](const SurfacePoint &, double) { return Vec2d::Zero(); }, {});
    std::string error;
    CHECK(sim.init(error));
    CHECK(sim.coarsening());
    for (int i = 0; i < 20; i++) sim.step();
    std::printf("  still object: %zu rebuilds, %zu vertices\n", sim.rebuildCount(), sim.triangulation().vertexCount());
    CHECK(sim.rebuildCount() == 0);
}

TEST(coarsen_noop_without_refinement)
{
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    StartRotatingStar(sim, 0, true);
    for (int i = 0; i < 30; i++) sim.step();
    CHECK(sim.rebuildCount() == 0);
    CHECK(sim.triangulation().vertexCount() == tri.vertexCount());
}

TEST(coarsen_leak_after_rebuild_keeps_region)
{
    // A gap opened right after a rebuild: the reconstruction leaks and the level set carried through the
    // rebuild (on the input vertices, extended to the new ones) keeps the region.
    const IntrinsicTriangulation tri = springls_test::DelaunaySurface(springls_test::Icosphere(4));
    IntrinsicSpringls sim(tri);
    StartRotatingStar(sim, 1, true);
    int steps = 0;
    while (sim.rebuildCount() == 0 && steps < 400) {
        sim.step();
        steps++;
    }
    CHECK(sim.rebuildCount() >= 1);
    const double h = sim.meanEdgeLength();
    const double before = Area(sim);
    const IntrinsicTriangulation &refined = sim.triangulation();
    const Vec3d x = refined.toExtrinsic(sim.springls().front().particle).position;
    std::vector<uint8_t> keep(sim.springls().size(), 1);
    size_t removed = 0;
    for (size_t s = 0; s < keep.size(); s++) {
        if ((refined.toExtrinsic(sim.springls()[s].particle).position - x).norm() < 3.0 * h) {
            keep[s] = 0;
            removed++;
        }
    }
    sim.removeSpringls(keep);
    sim.rebuildMesh();
    const size_t leaks = sim.leakCount();
    sim.step();
    const double after = Area(sim);
    std::printf("  leak after %d steps, %zu rebuilds: removed %zu springls, leaks %zu, area error %.5f\n", steps, sim.rebuildCount(), removed,
                sim.leakCount() - leaks, after / before - 1.0);
    CHECK(removed >= 8);
    CHECK(sim.leakCount() >= leaks + 1);
    CHECK(std::abs(after - before) < 0.02 * before);
}
