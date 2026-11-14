#include "Experiment.hpp"

#include <chrono>
#include <iomanip>

namespace enright
{
using namespace springls;

bool Experiment::init(const ExperimentOptions &options, std::string &error)
{
    mOptions = options;
    mRuns.clear();
    mStepCount = 0;
    if (mOptions.methods.empty()) {
        error = "no methods selected";
        return false;
    }
    for (Method method : mOptions.methods) {
        Run run;
        run.method = method;
        run.sim = MakeSimulation(method, mOptions.maxDepth);
        run.sim->verbose = mOptions.verbose;
        for (const auto &[key, value] : mOptions.tuning) {
            if (!SetTuning(*run.sim, key, value)) {
                error = "unknown tuning parameter '" + key + "'";
                return false;
            }
        }
        ConfigureEnright(*run.sim, mOptions.config);
        const auto start = std::chrono::steady_clock::now();
        if (!run.sim->init()) {
            error = std::string("failed to initialize ") + MethodTitle(method);
            return false;
        }
        run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        run.initial = MeasureEnright(*run.sim, mOptions.config);
        run.current = run.initial;
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

const char *PacingTitle(Pacing pacing)
{
    switch (pacing) {
        case Pacing::Lockstep: return "lockstep (one step each)";
        case Pacing::EqualCompute: return "equal compute (least wall-clock time steps next)";
        case Pacing::SameTime: return "same simulation time (furthest behind steps next)";
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
        run.current = MeasureEnright(*run.sim, mOptions.config);
        advanced = true;
    }
    if (advanced) {
        mStepCount++;
        writeCsvRows();
    }
    return advanced;
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
    mCsv << "method,step,time,area,relative_area_error,mean_shape_error,max_shape_error,springls,springls_primal,springls_dual,step_seconds,substeps\n";
    writeCsvRows();
    return true;
}

void Experiment::writeCsvRows()
{
    if (!mCsv.is_open()) return;
    mCsv << std::setprecision(9);
    for (const Run &run : mRuns) {
        if (!run.stepped) continue;  // finished earlier; its last row is already written
        const EnrightMetrics2D &m = run.current;
        mCsv << MethodKey(run.method) << ',' << run.sim->getIteration() << ',' << m.time << ',' << m.area << ',' << m.relativeAreaError << ','
             << m.meanShapeError << ',' << m.maxShapeError << ',' << m.springlCount << ',' << m.constellationSpringlCounts[0] << ','
             << m.constellationSpringlCounts[1] << ',' << run.stepSeconds << ',' << run.sim->getAdvectionSubstepCount() << '\n';
    }
    mCsv.flush();
}

void Experiment::printSummary(std::ostream &out) const
{
    const auto flags = out.flags();
    out << "Enright 2D, grid " << config().gridSize << ", T = " << config().period << ", errors at t = " << std::fixed << std::setprecision(3)
        << (mRuns.empty() ? 0.0 : mRuns.front().current.time) << " (domain units):\n";
    out << std::setprecision(5);
    for (const Run &run : mRuns) {
        const EnrightMetrics2D &m = run.current;
        out << "  " << std::setw(28) << std::left << MethodTitle(run.method) << std::right << " area error " << std::setw(9) << 100.0 * m.relativeAreaError
            << " %   mean shape " << m.meanShapeError << "   max shape " << m.maxShapeError << "   springls " << std::setw(6) << m.springlCount << "   steps "
            << run.sim->getIteration() << "   sub-steps " << run.sim->getAdvectionSubstepCount() << "   " << std::setprecision(2) << run.seconds << " s"
            << std::setprecision(5) << '\n';
    }
    out.flags(flags);
}
}  // namespace enright
