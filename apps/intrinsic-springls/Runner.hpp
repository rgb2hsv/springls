#pragma once
// intrinsic-springls: runs the closed-loop scenario (LoopScenario) with one region of springls on a
// triangle-mesh surface, adaptively refined near the springls, headless, writing per-step metrics (CSV)
// and frames (PLY).
#include <intrinsic/IntrinsicLoopScenario.hpp>
#include <intrinsic/IntrinsicReference.hpp>

#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace springls::intrinsic
{
// What the simulation runs on and how it is tuned.
struct SessionOptions {
    std::filesystem::path mesh;  // closed triangle mesh (PLY); empty: MakeIcosphere(5)
    LoopConfig config;
    // Adaptive refinement levels of the simulation's triangulation near the springls (IntrinsicSpringls::setMaxLevel).
    int maxLevel = IntrinsicSpringls::kDefaultMaxLevel;
    std::vector<std::pair<std::string, double>> tuning;  // SpringLevelSet2D tuning keys, lengths in h_t
    bool verbose = false;
};

// The closed-loop scenario with one region of springls on a mesh: load once, then step.
class Session
{
public:
    Session() = default;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    // False with a message when the mesh cannot be read, is not closed, or the scenario cannot be set up.
    bool load(const SessionOptions &options, std::string &error, std::ostream *log = nullptr);
    bool step();  // false once the loop is complete
    bool complete() const { return mComplete; }
    // The simulation's (adaptively refined) triangulation: every surface point of the simulation and the
    // reference lives on it. Before load, the input triangulation.
    const IntrinsicTriangulation &triangulation() const { return mSim ? mSim->triangulation() : mTri; }
    // The triangulation the scenario was set up on (the input mesh, Delaunay flipped: the loop distance
    // field, the object center).
    const IntrinsicTriangulation &inputTriangulation() const { return mTri; }
    const LoopScenario &scenario() const { return mScenario; }
    const IntrinsicSpringls &simulation() const { return *mSim; }
    // measure(), with the reference's three numbers filled in: each contour point's distance to the reference is
    // measured exactly in local charts, falling back to a lazily built fast march (capped at 20 h) only where
    // no chart reaches the reference.
    LoopScenario::Metrics metrics() const;
    const IntrinsicReference &reference() const { return mReference; }
    const std::vector<int> &objectLabels() const { return mLabels; }
    double lengthScale() const { return mSim ? mSim->meanEdgeLength() : 0.0; }  // the springls' length unit

private:
    IntrinsicTriangulation mTri;
    LoopScenario mScenario;
    std::unique_ptr<IntrinsicSpringls> mSim;
    IntrinsicReference mReference;
    std::vector<int> mLabels;
    bool mComplete = false;
};

// With --loop-distance 0, the runner places the object at this fraction of the scenario's automatic ρ
// (LoopConfig::loopFraction): see README.md for the measured cost.
constexpr double kDefaultLoopFraction = 0.125;

struct RunnerOptions {
    SessionOptions session;        // ParseArguments fills session.mesh with DefaultMesh() when --mesh is not given
    std::filesystem::path frames;  // directory for frame_NNNNNN.ply / springls_NNNNNN.ply (none if empty)
    std::filesystem::path csv;     // metrics file (none if empty)
    std::filesystem::path images;  // directory for image_NNNNNN.png views of the object (none if empty)
    int every = 1;                 // frames and images every this many steps (and at the last step)
    int imageSize = 512;
    int steps = -1;  // < 0: until the loop is complete
    int threads = 0;
};

struct RunnerResult {
    size_t vertices = 0, faces = 0;  // of the mesh the simulation ran on
    double lengthScale = 0.0;
    double loopDistance = 0.0, radius = 0.0;
    int steps = 0;
    double seconds = 0.0;
    LoopScenario::Metrics final;
};

// Parses the command line; false with a message (or the help text, with help set) otherwise.
bool ParseArguments(int argc, char **argv, RunnerOptions &options, std::string &error, bool *help = nullptr);
std::string DefaultMesh();
bool Run(const RunnerOptions &options, RunnerResult &result, std::string &error, std::ostream *log = nullptr);

// The CSV header, and one row per object label of the metrics after `step` (one: the region, label 1).
extern const char *const kCsvHeader;
void WriteCsvRows(std::ostream &out, int step, const LoopScenario::Metrics &metrics, const std::vector<int> &labels);
// frame_NNNNNN.ply (the input mesh, vertices colored inside / outside) and springls_NNNNNN.ply (each springl
// as two 3D edges through its particle, mapped to the input mesh).
bool WriteFrame(const std::filesystem::path &directory, int step, const IntrinsicSpringls &sim, std::string &error);
}  // namespace springls::intrinsic
