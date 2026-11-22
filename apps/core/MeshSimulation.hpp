#pragma once
// The mesh scenario (today's springls-viewer session): loads a mesh, fits an adaptive grid and runs
// DoubleWeave springls on it with no flow, so the surface only relaxes and resamples. No GL: load()
// and step() may run on a worker thread; call snapshot() from the same thread (or while no step runs).
#include "Simulation.hpp"

#include <io/Ply.hpp>
#include <levelset/IsoSurface3D.hpp>
#include <mesh/MeshLevelSet3D.hpp>
#include <spatial/AdaptiveGrid3D.hpp>
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace toybox
{
struct MeshStats {
    uint64_t steps = 0;
    double initSeconds = 0, lastStepSeconds = 0;
    size_t primal = 0, dual = 0, isoVertexes = 0, isoFaces = 0, leaves = 0;
    int finestDepth = 0;
    float finestCell = 0;
    springls::Vec3i baseCells = springls::Vec3i::Zero();
    bool clamped = false;
    bool fromCache = false;    // the initial state was restored from the cache
    int settleIterations = 0;  // relax / contract / fill iterations initialization ran
    int threads = 0;           // OpenMP threads in use on the thread that loaded the mesh
    size_t inputFaces = 0;
    springls::MeshTopology topology;
};

// Everything a renderer needs, copied out of the live simulation so it can be drawn while the next step runs.
struct MeshSnapshot {    // grid space (base voxels)
    uint64_t stamp = 0;  // new on every init / step, unique across simulations
    std::shared_ptr<const springls::AdaptiveGrid3D> grid;
    springls::Manifold3D manifold;
    springls::SurfaceMesh3D inputMesh;  // triangles, grid space, for the optional layer
    springls::MeshGridFit fit;
    MeshStats stats;
    std::vector<std::string> warnings;  // e.g. open mesh, clamped resolution
};

class MeshSimulation : public Simulation
{
public:
    // The registry's mesh entry; options for init() (the mesh, grid fit, cache, ...).
    MeshSimulation();
    MeshSimulation(const Entry &entry, const MeshOptions &options);

    // Load, fit and initialize. On failure returns false with a message in error and keeps the previous state.
    bool load(const MeshOptions &options, std::string &error);
    // load() with new options (mesh or grid edits, Reset in the window); they become this simulation's options.
    bool reload(const MeshOptions &options, std::string &error);
    void setResampling(bool on);
    MeshSnapshot snapshot() const;
    bool loaded() const { return mSim != nullptr; }
    const MeshOptions &options() const { return mOptions; }

    // Headless statistics: the initialization line and one line per step (wall-clock times included).
    void printInit(std::ostream &out) const;
    void printStep(std::ostream &out) const;

    const Entry &entry() const override { return mEntry; }
    bool init(std::string &error) override { return load(mOptions, error); }
    bool step() override { return stepOnce(); }
    bool complete() const override { return false; }
    double time() const override { return 0.0; }
    uint64_t iteration() const override { return mStats.steps; }
    uint64_t substeps() const override { return 0; }
    size_t springlCount() const override { return mStats.primal + mStats.dual; }
    bool supportsCsv() const override { return false; }
    void summary(std::ostream &out, double seconds) const override { printStep(out); }

private:
    bool stepOnce();
    void updateStats();

    const Entry &mEntry;
    MeshOptions mOptions;
    std::unique_ptr<springls::AdaptiveDweaveLevelSet3D> mSim;
    springls::SurfaceMesh3D mInputMesh;
    springls::MeshGridFit mFit;
    MeshStats mStats;
    std::vector<std::string> mWarnings;
    uint64_t mStamp = 0;
};
}  // namespace toybox
