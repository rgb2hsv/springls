#include "Check.hpp"

#include <core/Cli.hpp>
#include <core/Demos.hpp>
#include <core/Experiment.hpp>
#include <core/Registry.hpp>
#include <core/SphereSimulation.hpp>

#include <intrinsic-springls/Runner.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace toybox;

namespace
{
// Scratch files go in the system temp directory.
std::string Scratch() { return (std::filesystem::temp_directory_path() / "sphere-scenario").string(); }

// The sphere demo's invocation for the arguments, parsed as the command line would.
bool Parse(const std::vector<std::string> &args, Invocation &inv, std::string &error)
{
    const Demo *demo = demos::Find("sphere-surf-multi-springls");
    if (!demo) std::abort();
    std::vector<const char *> argv = {demo->program.c_str()};
    for (const std::string &a : args) argv.push_back(a.c_str());
    int exitCode = 0;
    return ParseInvocation(*demo, static_cast<int>(argv.size()), argv.data(), inv, error, exitCode);
}

SphereSimulation *Sphere(Experiment &experiment) { return experiment.runs().front().as<SphereSimulation>(); }
}  // namespace

TEST(sphere_headless_runs_and_writes_csv)
{
    Invocation inv;
    std::string error;
    const std::string csv = Scratch() + "/s.csv";
    CHECK(Parse({"--object", "star", "--radius", "0.4", "--steps", "3", "--no-gui", "--csv", csv}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    CHECK(experiment.openCsv(inv.csv));
    int steps = 0;
    while (experiment.step()) steps++;
    CHECK(steps == 3);
    experiment.closeCsv();
    std::ifstream in(csv);
    std::string line;
    std::getline(in, line);
    CHECK(line == springls::intrinsic::kCsvHeader);
    int rows = 0;
    while (std::getline(in, line)) rows++;
    CHECK(rows == 4);  // one region: the initial state and three steps
}

TEST(sphere_headless_reports_setup_error)
{
    Invocation inv;
    std::string error;
    // An object larger than the sphere: its outline does not enclose a region.
    CHECK(Parse({"--radius", "2.5", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(!experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    std::printf("  setup error: %s\n", error.c_str());
    CHECK(error.find("outline") != std::string::npos);
}

TEST(sphere_snapshot_counts)
{
    Invocation inv;
    std::string error;
    CHECK(Parse({"--radius", "0.4", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    SphereSimulation &sim = *Sphere(experiment);
    const SphereSnapshot snap = sim.snapshot();
    CHECK(snap.vertices.size() == sim.session().triangulation().vertexCount());
    CHECK(snap.labels.size() == snap.vertices.size());
    CHECK(snap.distances.size() == snap.vertices.size());
    for (float d : snap.distances) CHECK(std::isfinite(d) && d >= 0.0f);
    CHECK(!snap.triangles.empty());
    CHECK(!snap.springls.empty());
    CHECK(snap.springls.size() == sim.springlCount());
    CHECK(snap.springlLabels.size() == snap.springls.size());
    CHECK(!snap.contour.empty());
    CHECK(snap.contourLabels.size() == snap.contour.size());
    CHECK(!snap.arrows.empty());
    CHECK(!snap.reference.empty() && snap.reference.size() == sim.session().simulation().contour().size());
    for (const auto &polyline : snap.reference) {
        CHECK(!polyline.empty());
        for (const Vec3f &p : polyline) CHECK(std::abs(p.norm() - 1.0f) < 0.02f);
    }
    CHECK(snap.stamp != 0);
    CHECK(!snap.complete);
    CHECK(snap.period > 0 && snap.objectRadius > 0 && snap.lengthScale > 0);
    // The arrows sample the whole sphere (face centroids over the whole mesh), so they reach both the band the
    // object sweeps and beyond it: the unit sphere's geodesic distance from the loop center is the angle to it.
    const springls::Vec3 pole = snap.vertices[sim.session().scenario().center()];
    const double rho = sim.session().scenario().loopDistance(), r = snap.objectRadius, h = sim.session().inputTriangulation().meanEdgeLength();
    size_t inBand = 0;
    for (const SphereArrow &a : snap.arrows) {
        CHECK(std::abs(a.from.norm() - 1.0f) < 0.02f);
        const double d = std::acos(std::min(1.0f, a.from.normalized().dot(pole.normalized())));
        inBand += d >= rho - 2 * r - 2 * h - 0.05 && d <= rho + 2 * r + 2 * h + 0.05;
        CHECK(a.speed > 0);
    }
    CHECK(inBand > 0 && inBand < snap.arrows.size());
}

TEST(sphere_snapshot_uses_refined_mesh)
{
    Invocation inv;
    std::string error;
    CHECK(Parse({"--radius", "0.4", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    SphereSimulation &sim = *Sphere(experiment);
    const SphereSnapshot snap = sim.snapshot();
    const springls::IntrinsicTriangulation &tri = sim.session().triangulation();
    CHECK(snap.vertices.size() > sim.session().inputTriangulation().vertexCount());
    CHECK(snap.vertices.size() == tri.vertexCount() && snap.triangles.size() == tri.faceCount());
    const std::vector<double> &phi = sim.session().simulation().phi();
    CHECK(snap.labels.size() == phi.size() && snap.distances.size() == phi.size());
    size_t inside = 0;
    for (size_t v = 0; v < phi.size(); v++) {
        CHECK(snap.labels[v] == (phi[v] < 0.0 ? springls::IntrinsicSpringls::kRegionLabel : 0));
        CHECK(std::abs(snap.distances[v] - static_cast<float>(std::abs(phi[v]))) < 1e-6f);
        if (phi[v] < 0.0) inside++;
    }
    CHECK(inside > 0 && inside < phi.size());
    for (const auto &t : snap.triangles) CHECK(t[0] < snap.vertices.size() && t[1] < snap.vertices.size() && t[2] < snap.vertices.size());
    CHECK(snap.contractedTotal == sim.session().simulation().contractedTotal());
    CHECK(snap.filledTotal == sim.session().simulation().filledTotal());
}

TEST(sphere_arrows_independent_of_max_level)
{
    // Arrow spacing and length are in input edge lengths: refining does not change them.
    std::vector<SphereArrow> arrows[2];
    int k = 0;
    for (const char *level : {"0", "2"}) {
        Invocation inv;
        std::string error;
        CHECK(Parse({"--radius", "0.4", "--max-level", level, "--no-gui"}, inv, error));
        Experiment experiment;
        CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
        arrows[k++] = SampleArrows(Sphere(experiment)->session(), 2.0, 1.0);
    }
    CHECK(!arrows[0].empty() && arrows[0].size() == arrows[1].size());
    for (size_t i = 0; i < arrows[0].size() && i < arrows[1].size(); i++) {
        CHECK((arrows[0][i].from - arrows[1][i].from).norm() < 1e-6f);
        CHECK((arrows[0][i].to - arrows[1][i].to).norm() < 1e-4f);
    }
}

TEST(sphere_arrows_follow_schedule_sign)
{
    Invocation inv;
    std::string error;
    CHECK(Parse({"--speed", "50", "--radius", "0.4", "--max-level", "0", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    SphereSimulation &sim = *Sphere(experiment);
    const std::vector<SphereArrow> before = SampleArrows(sim.session(), 2.0, 1.0);
    const double half = sim.session().scenario().period() / 2;
    int guard = 0;
    while (sim.time() <= half && guard++ < 100000) CHECK(experiment.step());
    CHECK(sim.time() > half);
    const std::vector<SphereArrow> after = SampleArrows(sim.session(), 2.0, 1.0);
    CHECK(!before.empty());
    CHECK(before.size() == after.size());
    for (size_t i = 0; i < before.size() && i < after.size(); i++) {
        CHECK((before[i].from - after[i].from).norm() < 1e-6f);
        CHECK((before[i].to - before[i].from).dot(after[i].to - after[i].from) < 0.0f);
    }
}

TEST(sphere_steps_stop_at_period)
{
    Invocation inv;
    std::string error;
    CHECK(Parse({"--speed", "50", "--radius", "0.4", "--max-level", "0", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    SphereSimulation &sim = *Sphere(experiment);
    int guard = 0;
    while (sim.step() && guard++ < 100000) {
    }
    CHECK(guard < 100000);
    CHECK(sim.complete());
    CHECK(std::abs(sim.time() - sim.session().scenario().period()) < 1e-9);
    const uint64_t stamp = sim.snapshot().stamp, steps = sim.snapshot().steps;
    CHECK(!sim.step());
    CHECK(sim.snapshot().steps == steps);
    CHECK(sim.snapshot().stamp == stamp);
    CHECK(sim.snapshot().complete);
}

TEST(sphere_snapshot_with_no_springls)
{
    // Contracting every springl cannot empty a run (fill reseeds an empty region from its contour), so the
    // object is removed instead: re-initialize the library simulation with one object covering the whole surface (no contour).
    Invocation inv;
    std::string error;
    CHECK(Parse({"--radius", "0.4", "--max-level", "0", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    SphereSimulation &sim = *Sphere(experiment);
    springls::IntrinsicSpringls &library = const_cast<springls::IntrinsicSpringls &>(sim.session().simulation());
    const size_t n = sim.session().triangulation().vertexCount();
    library.setInitialRegion(std::vector<double>(n, -1.0));
    CHECK(library.init(error));
    CHECK(sim.springlCount() == 0);
    CHECK(sim.step());
    const SphereSnapshot snap = sim.snapshot();
    CHECK(snap.springls.empty());
    CHECK(snap.springlLabels.empty());
    CHECK(snap.contour.empty());
    CHECK(std::isfinite(snap.objectCenter.x()) && std::isfinite(snap.objectCenter.y()) && std::isfinite(snap.objectCenter.z()));
    CHECK(snap.objectCenter.norm() > 0.5f);
    CHECK(std::abs(snap.objectNormal.norm() - 1.0f) < 1e-4f);
    CHECK(std::isfinite(snap.metrics.meanShapeError) && std::isfinite(snap.metrics.maxShapeError));
}

TEST(sphere_headless_steps_zero_takes_no_step)
{
    Invocation inv;
    std::string error;
    CHECK(Parse({"--steps", "0", "--no-gui"}, inv, error));
    Experiment experiment;
    CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
    int steps = 0;
    while (experiment.step() && steps < 10) steps++;
    CHECK(steps == 0);
    CHECK(Sphere(experiment)->iteration() == 0 && Sphere(experiment)->snapshot().steps == 0);
}

TEST(sphere_object_center_is_region_centroid)
{
    // The star is symmetric about its center: the regions' centroid, on the surface, is the scenario's object center.
    for (const char *object : {"star", "puzzle"}) {
        Invocation inv;
        std::string error;
        CHECK(Parse({"--object", object, "--no-gui"}, inv, error));
        Experiment experiment;
        CHECK(experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error));
        SphereSimulation &sim = *Sphere(experiment);
        const SphereSnapshot snap = sim.snapshot();
        const springls::IntrinsicTriangulation &tri = sim.session().inputTriangulation();
        const Vec3f expected = tri.toExtrinsic(sim.session().scenario().objectCenter()).position.cast<float>();
        const float offset = (snap.objectCenter - expected).norm();
        std::cout << "  " << object << ": centroid " << offset / snap.lengthScale << " h from the object center" << std::endl;
        // The unit sphere: on the surface, the normal radial.
        CHECK(std::abs(snap.objectCenter.norm() - 1.0f) < 0.1f * snap.lengthScale);
        CHECK(snap.objectNormal.dot(snap.objectCenter.normalized()) > 0.999f);
        if (std::string(object) == "star") CHECK(offset < 0.5f * snap.lengthScale);
        // The puzzle piece's tabs move its centroid off the center, but well within the object.
        CHECK(offset < snap.objectRadius);
        CHECK(snap.bandWidth == static_cast<float>(springls::IntrinsicSpringls::kBandWidth) * snap.lengthScale);
    }
}
