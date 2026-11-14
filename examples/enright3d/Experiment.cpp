#include "Experiment.hpp"

#include <springls/AdaptiveSpringLevelSet3D.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>

namespace enright3d
{
using namespace springls;

bool Experiment::init(const ExperimentOptions &options, std::string &error)
{
    mOptions = options;
    mRuns.clear();
    mStepCount = 0;
    mGeneration++;
    if (mOptions.methods.empty()) {
        error = "no methods selected";
        return false;
    }
    if (!mOptions.resumeFile.empty() && mOptions.methods.size() != 1) {
        error = "resuming from a state needs exactly one method";
        return false;
    }
    std::sort(mOptions.checkpointTimes.begin(), mOptions.checkpointTimes.end());
    for (Method3D method : mOptions.methods) {
        Run run;
        run.method = method;
        run.config = mOptions.config;
        if (run.config.gridSize <= 0) run.config.gridSize = DefaultGridSize(method);
        run.sim = MakeSimulation(method, mOptions.elementType, mOptions.maxDepth);
        run.sim->verbose = mOptions.verbose;
        // Springl trajectories (correspondence and tracking) are not followed.
        run.sim->setTrackingEnabled(false);
        for (const auto &[key, value] : mOptions.tuning) {
            if (!SetTuning(*run.sim, key, value)) {
                error = "unknown tuning parameter '" + key + "'";
                return false;
            }
        }
        ConfigureEnright(*run.sim, run.config);
        const auto start = std::chrono::steady_clock::now();
        if (!run.sim->init()) {
            error = std::string("failed to initialize ") + MethodTitle(method);
            return false;
        }
        run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        run.initial = MeasureEnright(*run.sim, run.config);
        if (!mOptions.resumeFile.empty()) {
            std::ifstream in(mOptions.resumeFile, std::ios::binary);
            if (!in) {
                error = "could not open " + mOptions.resumeFile.string();
                return false;
            }
            std::string message;
            if (!run.sim->loadState(in, message)) {
                error = mOptions.resumeFile.string() + ": " + message;
                return false;
            }
            std::cout << "Resumed " << MethodTitle(method) << " from " << mOptions.resumeFile.string() << " at t = " << run.sim->getSimulationTime()
                      << " (step " << run.sim->getIteration() << ")" << std::endl;
        }
        run.current = MeasureEnright(*run.sim, run.config);
        run.startSpringls = run.current.springlCount;
        run.done = run.sim->isScheduleComplete() || run.current.time >= mOptions.stopAt || mOptions.maxSteps == 0;
        // Checkpoint times already passed (when resuming) are not written again.
        while (run.nextCheckpoint < mOptions.checkpointTimes.size() && mOptions.checkpointTimes[run.nextCheckpoint] < run.current.time) run.nextCheckpoint++;
        mRuns.push_back(std::move(run));
    }
    writeCsvRows();
    return true;
}

const char *PacingKey(Pacing pacing)
{
    switch (pacing) {
        case Pacing::Lockstep: return "lockstep";
        case Pacing::EqualCompute: return "equal-compute";
        case Pacing::SameTime: return "same-time";
    }
    return "";
}

bool ParsePacing(const std::string &key, Pacing &pacing)
{
    for (Pacing p : {Pacing::Lockstep, Pacing::EqualCompute, Pacing::SameTime}) {
        if (key == PacingKey(p)) {
            pacing = p;
            return true;
        }
    }
    return false;
}

bool Experiment::step()
{
    // The one run to advance under EqualCompute / SameTime (ties go to the earlier run).
    const Run *next = nullptr;
    if (mOptions.pacing != Pacing::Lockstep) {
        for (const Run &run : mRuns) {
            if (run.done) continue;
            const auto key = [&](const Run &r) { return mOptions.pacing == Pacing::EqualCompute ? r.seconds : r.sim->getSimulationTime(); };
            if (!next || key(run) < key(*next)) next = &run;
        }
    }
    bool advanced = false;
    for (Run &run : mRuns) {
        run.stepped = !run.done && (mOptions.pacing == Pacing::Lockstep || &run == next);
        if (!run.stepped) continue;
        const auto start = std::chrono::steady_clock::now();
        run.done = !run.sim->step();
        run.stepSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        run.seconds += run.stepSeconds;
        run.current = MeasureEnright(*run.sim, run.config);
        run.stepsTaken++;
        if (run.current.time >= mOptions.stopAt || run.stepsTaken >= mOptions.maxSteps) run.done = true;
        if (static_cast<double>(run.current.springlCount) > mOptions.maxSpringlGrowth * static_cast<double>(run.startSpringls)) {
            run.done = run.aborted = true;
            std::cout << "Aborted " << MethodTitle(run.method) << " at t = " << run.current.time << " (step " << run.sim->getIteration()
                      << "): springls grew from " << run.startSpringls << " to " << run.current.springlCount << std::endl;
        }
        std::string error;
        if (!saveCheckpoints(run, error)) std::cerr << "error: " << error << std::endl;
        advanced = true;
    }
    if (advanced) {
        mStepCount++;
        writeCsvRows();
    }
    return advanced;
}

bool Experiment::saveCheckpoints(Run &run, std::string &error)
{
    while (run.nextCheckpoint < mOptions.checkpointTimes.size() && run.current.time >= mOptions.checkpointTimes[run.nextCheckpoint]) {
        const double t = mOptions.checkpointTimes[run.nextCheckpoint++];
        std::filesystem::create_directories(mOptions.checkpointDir);
        char name[160];
        std::snprintf(name, sizeof(name), "%s_%s_g%d_d%d_t%.4f.state", MethodKey(run.method), ElementTypeKey(mOptions.elementType), run.config.gridSize,
                      mOptions.maxDepth, t);
        const std::filesystem::path file = mOptions.checkpointDir / name;
        std::ofstream out(file, std::ios::binary);
        if (!out || !run.sim->saveState(out, error)) {
            if (error.empty()) error = "could not write " + file.string();
            return false;
        }
        std::cout << "Saved " << MethodTitle(run.method) << " at t = " << run.current.time << " (step " << run.sim->getIteration() << ") to " << file.string()
                  << std::endl;
    }
    return true;
}

ExperimentSnapshot Experiment::snapshot() const
{
    ExperimentSnapshot s;
    s.period = mOptions.config.period;
    s.generation = mGeneration;
    s.runs.reserve(mRuns.size());
    for (const Run &run : mRuns) {
        RunSnapshot r;
        r.method = run.method;
        r.config = run.config;
        if (const auto *adaptive = dynamic_cast<const AdaptiveSpringLevelSet3D *>(run.sim.get())) {
            r.finestDepth = adaptive->getGrid().finestDepth();
            r.maxDepth = adaptive->getGrid().maxDepth();
            r.leafCount = adaptive->getGrid().leaves().size();
            r.grid = std::make_shared<const AdaptiveGrid3D>(adaptive->getGrid());
        } else {
            r.levelSet = std::make_shared<const VolumeF>(run.sim->getLevelSet());
        }
        r.manifold = run.sim->getManifold();
        r.metrics = run.current;
        r.iteration = run.sim->getIteration();
        r.substeps = run.sim->getAdvectionSubstepCount();
        r.done = run.done;
        r.seconds = run.seconds;
        s.runs.push_back(std::move(r));
    }
    return s;
}

bool Experiment::done() const
{
    for (const Run &run : mRuns) {
        if (!run.done) return false;
    }
    return true;
}

bool Experiment::openCsv(const std::filesystem::path &file)
{
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path());
    mCsv.open(file);
    if (!mCsv) return false;
    mCsv << "method,elements,grid,step,time,volume,relative_volume_error,mean_shape_error,max_shape_error,springls,springls_primal,springls_dual,step_seconds,"
            "substeps\n";
    writeCsvRows();
    return true;
}

void Experiment::writeCsvRows()
{
    if (!mCsv.is_open()) return;
    mCsv << std::setprecision(9);
    for (const Run &run : mRuns) {
        if (!run.stepped) continue;  // finished earlier; its last row is already written
        const EnrightMetrics3D &m = run.current;
        mCsv << MethodKey(run.method) << ',' << ElementTypeKey(mOptions.elementType) << ',' << run.config.gridSize << ',' << run.sim->getIteration() << ','
             << m.time << ',' << m.volume << ',' << m.relativeVolumeError << ',' << m.meanShapeError << ',' << m.maxShapeError << ',' << m.springlCount << ','
             << m.constellationSpringlCounts[0] << ',' << m.constellationSpringlCounts[1] << ',' << run.stepSeconds << ','
             << run.sim->getAdvectionSubstepCount() << '\n';
    }
    mCsv.flush();
}

void Experiment::printSummary(std::ostream &out) const
{
    const auto flags = out.flags();
    out << "Enright 3D, " << ElementTypeKey(mOptions.elementType) << " springls, T = " << period() << ", errors at t = " << std::fixed << std::setprecision(3)
        << (mRuns.empty() ? 0.0 : mRuns.front().current.time) << " (domain units):\n";
    out << std::setprecision(5);
    for (const Run &run : mRuns) {
        const EnrightMetrics3D &m = run.current;
        out << "  " << std::setw(31) << std::left << MethodTitle(run.method) << std::right << " grid " << std::setw(3) << run.config.gridSize
            << " volume error " << std::setw(9) << 100.0 * m.relativeVolumeError << " %   mean shape " << m.meanShapeError << "   max shape " << m.maxShapeError
            << "   springls " << std::setw(7) << m.springlCount << "   steps " << run.sim->getIteration() << "   sub-steps "
            << run.sim->getAdvectionSubstepCount() << "   " << std::setprecision(2) << run.seconds << " s" << std::setprecision(5) << '\n';
    }
    out.flags(flags);
}
}  // namespace enright3d
