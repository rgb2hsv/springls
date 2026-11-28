#include "Enright3DSimulation.hpp"

#include "Experiment.hpp"

#include <lod/MeshTracking3D.hpp>
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace toybox
{
using namespace springls;

Enright3DSimulation::Enright3DSimulation(const Entry &entry, const RunOptions &options) : mEntry(entry), mOptions(options)
{
    mConfig.gridSize = options.gridSize > 0 ? options.gridSize : DefaultGridSize(method());
    mConfig.period = options.period;
    mConfig.maxTimeStepFraction = options.maxTimeStepFraction;
}

bool Enright3DSimulation::init(std::string &error)
{
    mSim = MakeSimulation(method(), mOptions.elementType, mOptions.maxDepth);
    mSim->verbose = mOptions.verbose;
    // Springl trajectories (correspondence and tracking) are not followed.
    mSim->setTrackingEnabled(false);
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
    mComplete = mSim->isScheduleComplete();
    return true;
}

bool Enright3DSimulation::resume(const std::filesystem::path &stateFile, std::string &error)
{
    // The state file and its level sets beside it (<name>.state and <name>.vdb).
    std::filesystem::path levelSetFile = stateFile;
    levelSetFile.replace_extension(".vdb");
    std::ifstream in(stateFile, std::ios::binary), levelSets(levelSetFile, std::ios::binary);
    if (!in || !levelSets) {
        error = "could not open " + (!in ? stateFile : levelSetFile).string();
        return false;
    }
    std::string message;
    if (!mSim->loadState(in, levelSets, message)) {
        error = stateFile.string() + ": " + message;
        return false;
    }
    std::cout << "Resumed " << MethodTitle(method()) << " from " << stateFile.string() << " at t = " << mSim->getSimulationTime() << " (step "
              << mSim->getIteration() << ")" << std::endl;
    mCurrent = MeasureEnright(*mSim, mConfig);
    mComplete = mSim->isScheduleComplete();
    return true;
}

bool Enright3DSimulation::step()
{
    mComplete = !mSim->step();
    mCurrent = MeasureEnright(*mSim, mConfig);
    return !mComplete;
}

bool Enright3DSimulation::saveCheckpoint(const std::filesystem::path &dir, double t, std::string &error) const
{
    std::filesystem::create_directories(dir);
    char name[160];
    std::snprintf(name, sizeof(name), "%s_%s_g%d_d%d_t%.4f.state", MethodKey(method()), ElementTypeKey(mOptions.elementType), mConfig.gridSize,
                  mOptions.maxDepth, t);
    // The springls and scalar state (cereal) and the level sets (OpenVDB) beside it.
    const std::filesystem::path file = dir / name;
    std::filesystem::path levelSetFile = file;
    levelSetFile.replace_extension(".vdb");
    std::ofstream out(file, std::ios::binary), levelSets(levelSetFile, std::ios::binary);
    if (!out || !levelSets || !mSim->saveState(out, levelSets, error)) {
        if (error.empty()) error = "could not write " + (!out ? file : levelSetFile).string();
        return false;
    }
    std::cout << "Saved " << MethodTitle(method()) << " at t = " << mCurrent.time << " (step " << mSim->getIteration() << ") to " << file.string() << " and "
              << levelSetFile.filename().string() << std::endl;
    return true;
}

void Enright3DSimulation::csvHeader(std::ostream &out) const
{
    out << "method,elements,grid,step,time,volume,relative_volume_error,mean_shape_error,max_shape_error,springls,springls_primal,springls_dual,step_seconds,"
           "substeps\n";
}

void Enright3DSimulation::csvRow(std::ostream &out, double stepSeconds) const
{
    const EnrightMetrics3D &m = mCurrent;
    out << std::setprecision(9);
    out << MethodKey(method()) << ',' << ElementTypeKey(mOptions.elementType) << ',' << mConfig.gridSize << ',' << mSim->getIteration() << ',' << m.time << ','
        << m.volume << ',' << m.relativeVolumeError << ',' << m.meanShapeError << ',' << m.maxShapeError << ',' << m.springlCount << ','
        << m.constellationSpringlCounts[0] << ',' << m.constellationSpringlCounts[1] << ',' << stepSeconds << ',' << mSim->getAdvectionSubstepCount() << '\n';
}

void Enright3DSimulation::summaryHeader(std::ostream &out, double period) const
{
    out << "Enright 3D, " << ElementTypeKey(mOptions.elementType) << " springls, T = " << period << ", errors at t = " << std::fixed << std::setprecision(3)
        << mCurrent.time << " (domain units):\n";
}

void Enright3DSimulation::summary(std::ostream &out, double seconds) const
{
    const EnrightMetrics3D &m = mCurrent;
    out << "  " << std::setw(31) << std::left << MethodTitle(method()) << std::right << " grid " << std::setw(3) << mConfig.gridSize << " volume error "
        << std::setw(9) << 100.0 * m.relativeVolumeError << " %   mean shape " << m.meanShapeError << "   max shape " << m.maxShapeError << "   springls "
        << std::setw(7) << m.springlCount << "   steps " << mSim->getIteration() << "   sub-steps " << mSim->getAdvectionSubstepCount() << "   "
        << std::setprecision(2) << seconds << " s" << std::setprecision(5);
    if (const auto *tracking = dynamic_cast<const MeshTracking3D *>(mSim.get()); tracking && tracking->meshCheck()) {
        const MeshCheck &check = *tracking->meshCheck();
        out << "   fold-overs " << check.foldedEdges << " (most " << tracking->maxFoldedEdges() << ")   overlaps " << check.overlappingPairs << " (most "
            << tracking->maxOverlappingPairs() << ")";
    }
    out << '\n';
}

void Enright3DSimulation::progress(std::ostream &out, bool withTime) const
{
    if (withTime) {
        out << "  " << MethodKey(method()) << ": t " << mCurrent.time << ", " << mSim->getAdvectionSubstepCount() << " sub-steps, volume error "
            << 100.0 * mCurrent.relativeVolumeError << " %";
    } else {
        out << "  " << MethodKey(method()) << ": " << mCurrent.springlCount << " springls, volume error " << 100.0 * mCurrent.relativeVolumeError << " %";
    }
}

RunSnapshot Enright3DSimulation::snapshot() const
{
    RunSnapshot r;
    r.method = method();
    r.config = mConfig;
    if (const auto *adaptive = dynamic_cast<const AdaptiveSpringLevelSet3D *>(mSim.get())) {
        r.finestDepth = adaptive->getGrid().finestDepth();
        r.maxDepth = adaptive->getGrid().maxDepth();
        r.leafCount = adaptive->getGrid().leaves().size();
        r.grid = std::make_shared<const AdaptiveGrid3D>(adaptive->getGrid());
    } else {
        r.levelSet = std::make_shared<const VolumeF>(mSim->getLevelSet());
    }
    r.manifold = mSim->getManifold();
    if (const auto *tracking = dynamic_cast<const MeshTracking3D *>(mSim.get())) {
        r.trackedMesh = std::make_shared<const TriangleSurface>(tracking->mesh());
        r.trackedCheck = tracking->meshCheck();
    }
    r.metrics = mCurrent;
    r.iteration = mSim->getIteration();
    r.substeps = mSim->getAdvectionSubstepCount();
    return r;
}

ExperimentSnapshot Snapshot3D(const Experiment &experiment)
{
    ExperimentSnapshot s;
    s.period = experiment.period();
    s.generation = experiment.generation();
    s.runs.reserve(experiment.runs().size());
    for (const Run &run : experiment.runs()) {
        const auto *sim = run.as<Enright3DSimulation>();
        if (!sim) continue;
        RunSnapshot r = sim->snapshot();
        r.done = run.done;
        r.seconds = run.seconds;
        s.runs.push_back(std::move(r));
    }
    return s;
}
}  // namespace toybox
