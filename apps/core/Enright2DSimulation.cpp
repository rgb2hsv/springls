#include "Enright2DSimulation.hpp"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace toybox
{
using namespace springls;

Enright2DSimulation::Enright2DSimulation(const Entry &entry, const RunOptions &options) : mEntry(entry), mOptions(options)
{
    mConfig.gridSize = options.gridSize > 0 ? options.gridSize : DefaultGridSize(method());
    mConfig.period = options.period;
    mConfig.maxTimeStepFraction = options.maxTimeStepFraction;
}

bool Enright2DSimulation::init(std::string &error)
{
    mSim = MakeSimulation(method(), mOptions.maxDepth);
    if (!mSim) {
        error = std::string(MethodTitle(method())) + " is not implemented yet";
        return false;
    }
    mSim->verbose = mOptions.verbose;
    if (mOptions.substepsPerStep > 0) SetTuning(*mSim, "substepsPerStep", mOptions.substepsPerStep);
    for (const auto &[key, value] : mOptions.tuning) {
        if (!SetTuning(*mSim, key, value)) {
            error = "unknown tuning parameter '" + key + "'";
            return false;
        }
    }
    ConfigureEnright(*mSim, mConfig);
    if (!mSim->init()) {
        error = std::string("failed to initialize ") + MethodTitle(method());
        return false;
    }
    mInitial = MeasureEnright(*mSim, mConfig);
    mCurrent = mInitial;
    mComplete = false;
    return true;
}

bool Enright2DSimulation::resume(const std::filesystem::path &stateFile, std::string &error)
{
    std::ifstream in(stateFile, std::ios::binary);
    if (!in) {
        error = "could not open " + stateFile.string();
        return false;
    }
    std::string message;
    if (!mSim->loadState(in, message)) {
        error = stateFile.string() + ": " + message;
        return false;
    }
    std::cout << "Resumed " << MethodTitle(method()) << " from " << stateFile.string() << " at t = " << mSim->getSimulationTime() << " (step "
              << mSim->getIteration() << ")" << std::endl;
    mCurrent = MeasureEnright(*mSim, mConfig);
    mComplete = mSim->isScheduleComplete();
    return true;
}

bool Enright2DSimulation::saveCheckpoint(const std::filesystem::path &dir, double t, std::string &error) const
{
    std::filesystem::create_directories(dir);
    char name[160];
    std::snprintf(name, sizeof(name), "%s_g%d_d%d_t%.4f.state", MethodKey(method()), mConfig.gridSize, mOptions.maxDepth, t);
    const std::filesystem::path file = dir / name;
    std::ofstream out(file, std::ios::binary);
    if (!out || !mSim->saveState(out, error)) {
        if (error.empty()) error = "could not write " + file.string();
        return false;
    }
    std::cout << "Saved " << MethodTitle(method()) << " at t = " << mCurrent.time << " (step " << mSim->getIteration() << ") to " << file.string() << std::endl;
    return true;
}

bool Enright2DSimulation::step()
{
    mComplete = !mSim->step();
    mCurrent = MeasureEnright(*mSim, mConfig);
    return !mComplete;
}

void Enright2DSimulation::csvHeader(std::ostream &out) const
{
    out << "method,step,time,area,relative_area_error,mean_shape_error,max_shape_error,springls,springls_primal,springls_dual,step_seconds,substeps\n";
}

void Enright2DSimulation::csvRow(std::ostream &out, double stepSeconds) const
{
    const EnrightMetrics2D &m = mCurrent;
    out << std::setprecision(9);
    out << MethodKey(method()) << ',' << mSim->getIteration() << ',' << m.time << ',' << m.area << ',' << m.relativeAreaError << ',' << m.meanShapeError << ','
        << m.maxShapeError << ',' << m.springlCount << ',' << m.constellationSpringlCounts[0] << ',' << m.constellationSpringlCounts[1] << ',' << stepSeconds
        << ',' << mSim->getAdvectionSubstepCount() << '\n';
}

void Enright2DSimulation::labelsCsvHeader(std::ostream &out) const { out << "method,step,time,label,area,relative_area_error\n"; }

void Enright2DSimulation::labelsCsvRows(std::ostream &out) const
{
    const EnrightMetrics2D &m = mCurrent;
    out << std::setprecision(9);
    for (size_t l = 0; l < m.labelAreas.size(); l++) {
        out << MethodKey(method()) << ',' << mSim->getIteration() << ',' << m.time << ',' << (l + 1) << ',' << m.labelAreas[l] << ','
            << m.labelRelativeAreaErrors[l] << '\n';
    }
}

void Enright2DSimulation::summaryHeader(std::ostream &out, double period) const
{
    out << "Enright 2D, T = " << period << ", errors at t = " << std::fixed << std::setprecision(3) << mCurrent.time << " (domain units):\n";
}

void Enright2DSimulation::summary(std::ostream &out, double seconds) const
{
    const EnrightMetrics2D &m = mCurrent;
    out << "  " << std::setw(30) << std::left << MethodTitle(method()) << std::right << " grid " << std::setw(4) << mConfig.gridSize << "   area error "
        << std::setw(9) << 100.0 * m.relativeAreaError << " %   mean shape " << m.meanShapeError << "   max shape " << m.maxShapeError << "   springls "
        << std::setw(6) << m.springlCount << "   steps " << mSim->getIteration() << "   sub-steps " << mSim->getAdvectionSubstepCount() << "   "
        << std::setprecision(2) << seconds << " s" << std::setprecision(5) << '\n';
    if (!m.labelRelativeAreaErrors.empty()) {
        out << "  " << std::setw(30) << "" << "             region area errors";
        for (double e : m.labelRelativeAreaErrors) out << ' ' << std::setw(9) << 100.0 * e << " %";
        out << '\n';
    }
}
}  // namespace toybox
