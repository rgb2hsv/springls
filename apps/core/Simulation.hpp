#pragma once
// One run of one implemented configuration, type-erased over domain and scenario. The adapters
// (Enright2DSimulation, Enright3DSimulation, MeshSimulation) own the library simulation; windows
// downcast to the adapter of their domain to draw it.
#include "Registry.hpp"

#include <core/Common3D.hpp>
#include <mesh/MeshLevelSet3D.hpp>

#include <intrinsic-springls/Runner.hpp>

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace toybox
{
// The mesh scenario's options (today's springls-viewer settings).
struct MeshOptions {
    std::filesystem::path mesh;
    int threads = 0;  // OpenMP threads for the thread that runs the simulation (0: default)
    springls::MeshGridOptions grid;
    springls::ElementType elementType = springls::ElementType::Quad;
    std::vector<std::pair<std::string, double>> tuning;  // see springls::TuningKeys3D()
    // Directory of cached initial states (empty: no cache). load() restores the state after init() from it when
    // it holds one for this mesh and these settings, and otherwise writes one there after init().
    std::filesystem::path cacheDir;
    // Initialization ends with up to this many relax / contract / fill iterations (stopping once an iteration
    // removes and adds at most 0.01% of the springls); see SpringLevelSet3D::settle().
    int settleIterations = 20;
};

// The sphere scenario's options: multi-object springls on a triangle-mesh surface.
struct SphereOptions {
    springls::intrinsic::SessionOptions session;  // mesh empty unless given (the built-in icosphere)
    int every = 1;                                // frames every N steps (and at the end)
    double frameTime = 0.0;                       // > 0: frames each time the simulation time passes a multiple of this, instead
    std::vector<std::string> hidden;  // of "regions", "isolines", "springls", "particles", "vertices", "normals", "contour", "arrows", "reference", "wireframe"
    int threads = 0;
};

// Options of one run, resolved by the command line (defaults come from the profile).
struct RunOptions {
    int gridSize = 0;  // 0: the method's default
    double period = 8.0;
    double maxTimeStepFraction = 1.0 / 800.0;
    int maxDepth = 4;
    int substepsPerStep = 0;  // > 0: sets the substepsPerStep tuning before the --set overrides; 0: the method's default
    springls::ElementType elementType = springls::ElementType::Quad;  // 3D and mesh
    std::vector<std::pair<std::string, double>> tuning;
    bool verbose = false;
    MeshOptions mesh;      // mesh scenario only (its element type and tuning come from the fields above)
    SphereOptions sphere;  // sphere scenario only
};

// A stamp for a snapshot: new on every call, so a view tells a rebuilt simulation from the one before.
uint64_t NextSnapshotStamp();

class Simulation
{
public:
    virtual ~Simulation() = default;
    virtual const Entry &entry() const = 0;
    // Creates and initializes the library simulation; false with a message on failure.
    virtual bool init(std::string &error) = 0;
    virtual bool step() = 0;  // false once the scenario is complete
    virtual bool complete() const = 0;
    virtual double time() const = 0;
    virtual uint64_t iteration() const = 0;
    virtual uint64_t substeps() const = 0;
    virtual size_t springlCount() const = 0;

    // Metrics CSV in the scenario's format (header and one row per state).
    virtual bool supportsCsv() const = 0;
    virtual void csvHeader(std::ostream &) const {}
    virtual void csvRow(std::ostream &, double stepSeconds) const {}
    // Per-region rows of multi-object runs (<csv stem>-labels.csv).
    virtual bool hasLabelsCsv() const { return false; }
    virtual void labelsCsvHeader(std::ostream &) const {}
    virtual void labelsCsvRows(std::ostream &) const {}

    // Summary: the header line of the domain (written once, by the first run) and this run's line(s).
    virtual void summaryHeader(std::ostream &, double period) const {}
    virtual void summary(std::ostream &, double seconds) const = 0;
    // Headless progress: this run's part of the progress line (withTime: the semi-Lagrangian
    // programs' variant, with the run's time and sub-steps).
    virtual void progress(std::ostream &, bool withTime) const {}

    // Checkpoints: the state saved to dir once the run reaches time t, as <key>_<elements>_g<grid>_d<depth>_t<t>.state + .vdb.
    virtual bool supportsCheckpoints() const { return false; }
    virtual bool saveCheckpoint(const std::filesystem::path &dir, double t, std::string &error) const { return false; }
    virtual bool resume(const std::filesystem::path &stateFile, std::string &error) { return false; }
};
}  // namespace toybox
