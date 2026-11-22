#include "Main.hpp"

#include "MeshSimulation.hpp"
#include "SphereSimulation.hpp"

#include <scenarios/EnrightTest2D.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace toybox
{
namespace
{
int RunEnrightHeadless(const Invocation &invocation, Experiment &experiment)
{
    // Progress about every 10% of the period, by the run furthest behind.
    constexpr int kProgressReports = 10;
    const double period = experiment.period();
    int nextReport = 1;
    while (experiment.step()) {
        double t = period;
        for (const Run &run : experiment.runs()) {
            if (!run.done) t = std::min(t, run.sim->time());
        }
        if (t >= nextReport * period / kProgressReports) {
            std::cout << "  t = " << t << " / " << period << " (step " << experiment.stepCount() << ")";
            for (const Run &run : experiment.runs()) run.sim->progress(std::cout, invocation.slComparison);
            std::cout << std::endl;
            nextReport = static_cast<int>(std::floor(t / period * kProgressReports)) + 1;
        }
    }
    experiment.printSummary(std::cout);
    const auto tracking = [&](size_t i) { return experiment.runs()[i].sim->entry().config.tracking; };
    if (invocation.slComparison && experiment.runs().size() == 2 && tracking(0) == InterfaceTracking::Eulerian &&
        tracking(1) == InterfaceTracking::SemiLagrangian) {
        const Run &cfl = experiment.runs()[0];
        const Run &sl = experiment.runs()[1];
        const uint64_t cflSteps = cfl.sim->substeps(), slSteps = sl.sim->substeps();
        if (slSteps > 0 && sl.seconds > 0.0) {
            std::cout << "Semi-Lagrangian: " << static_cast<double>(cflSteps) / static_cast<double>(slSteps) << "x fewer advection sub-steps, "
                      << cfl.seconds / sl.seconds << "x less compute than the CFL-limited run" << std::endl;
        }
    }
    return 0;
}

int RunMeshHeadless(Experiment &experiment)
{
    const auto *mesh = experiment.runs().front().as<MeshSimulation>();
    for (const std::string &w : mesh->snapshot().warnings) std::cerr << "warning: " << w << std::endl;
    mesh->printInit(std::cout);
    mesh->printStep(std::cout);
    while (experiment.step()) mesh->printStep(std::cout);
    return 0;
}

int RunSphereHeadless(Experiment &experiment)
{
    const auto *sphere = experiment.runs().front().as<SphereSimulation>();
    sphere->printInit(std::cout);
    while (experiment.step()) sphere->printStep(std::cout);
    experiment.printSummary(std::cout);
    return 0;
}

// The line the old programs printed before running.
void PrintBanner(const Invocation &inv)
{
    const RunOptions &run = inv.run;
    if (inv.slComparison) {
        if (inv.domain == Domain::D2) {
            std::cout << (inv.slOnly ? "Semi-Lagrangian DoubleWeave" : "Semi-Lagrangian vs CFL-limited DoubleWeave") << ": Enright " << run.gridSize
                      << "^2, T = " << run.period << ", depth " << run.maxDepth << ", sub-step cap T * " << run.maxTimeStepFraction;
        } else {
            std::cout << (inv.slOnly ? "Semi-Lagrangian DoubleWeave 3D" : "Semi-Lagrangian vs CFL-limited DoubleWeave 3D") << ": Enright " << run.gridSize
                      << "^3 base, depth " << run.maxDepth << ", T = " << run.period << ", sub-step cap T * " << run.maxTimeStepFraction;
        }
        if (!inv.slOnly) std::cout << ", pacing " << PacingKey(inv.pacing);
        std::cout << ", " << springls::MaxThreads() << " thread(s)" << std::endl;
    } else if (inv.domain == Domain::D2) {
        std::cout << "Enright 2D: " << inv.methodLabel << ", grid "
                  << (run.gridSize > 0 ? std::to_string(run.gridSize) : std::string("256 uniform / 32 adaptive")) << ", T = " << run.period << ", max depth "
                  << run.maxDepth << ", " << springls::MaxThreads() << " thread(s)" << std::endl;
    } else {
        std::cout << "Enright 3D: " << inv.methodLabel << ", " << springls::ElementTypeKey(run.elementType) << " springls, T = " << run.period << ", max depth "
                  << run.maxDepth << ", " << springls::MaxThreads() << " thread(s)" << std::endl;
    }
}
}  // namespace

int RunHeadless(const Invocation &invocation, Experiment &experiment)
{
    if (invocation.scenario == Scenario::Mesh) return RunMeshHeadless(experiment);
    if (Surface(invocation.scenario)) return RunSphereHeadless(experiment);
    return RunEnrightHeadless(invocation, experiment);
}

int Main(int argc, char *argv[], const Demo &demo, WindowRunner window)
{
    Invocation inv;
    std::string error;
    int exitCode = 2;
    if (!ParseInvocation(demo, argc, argv, inv, error, exitCode)) {
        std::cerr << error << std::endl;
        return exitCode;
    }
    if (inv.list) {
        std::cout << Registry::Instance().markdownMatrix();
        return 0;
    }
    if (inv.listTuning) {
        for (const std::string &key : inv.domain == Domain::D3 ? springls::TuningKeys3D() : springls::TuningKeys()) std::cout << key << '\n';
        return 0;
    }
    for (const std::string &note : inv.notes) std::cerr << "note: " << note << std::endl;
    springls::SetThreads(inv.threads);  // a window's worker thread sets it again for itself

    Experiment experiment;
    if (Surface(inv.scenario)) {
        if (!inv.noGui) {
            if (window) return window(inv, experiment);
            std::cerr << "error: this build has no GUI; pass --no-gui" << std::endl;
            return 2;
        }
        if (!inv.frames.empty()) {
            if (window) return window(inv, experiment);
            std::cerr << "error: --frames needs the GUI build (SPRINGLS_BUILD_GUI=ON)" << std::endl;
            return 1;
        }
        if (!experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error)) {
            std::cerr << "error: " << error << std::endl;
            return 1;
        }
        if (!inv.csv.empty() && !experiment.openCsv(inv.csv)) {
            std::cerr << "error: could not write " << inv.csv.string() << std::endl;
            return 1;
        }
        return RunHeadless(inv, experiment);
    }
    if (inv.scenario == Scenario::Mesh) {
        if (!inv.noGui) {
            if (window) return window(inv, experiment);
            std::cerr << "error: this build has no GUI; pass --no-gui" << std::endl;
            return 2;
        }
        if (!experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error)) {
            std::cerr << "error: " << error << std::endl;
            return 1;
        }
        return RunHeadless(inv, experiment);
    }

    if (!experiment.init(inv.entries, inv.run, inv.pacing, inv.limits, error)) {
        std::cerr << "error: " << error << std::endl;
        return 1;
    }
    if (!inv.csv.empty() && !experiment.openCsv(inv.csv)) {
        std::cerr << "error: could not write " << inv.csv.string() << std::endl;
        return 1;
    }
    PrintBanner(inv);
    if (window) {
        if (!inv.noGui || !inv.frames.empty()) return window(inv, experiment);
    } else {
        if (!inv.noGui) std::cout << "(built without the GUI; running headless)" << std::endl;
        if (!inv.frames.empty()) {
            std::cerr << "error: --frames needs the GUI build (SPRINGLS_BUILD_GUI=ON)" << std::endl;
            return 1;
        }
    }
    return RunHeadless(inv, experiment);
}
}  // namespace toybox
