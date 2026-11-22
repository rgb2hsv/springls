#include "MeshSimulation.hpp"

#include <core/Volume.hpp>
#include <mesh/MeshDistance3D.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace springls;

namespace toybox
{
uint64_t NextSnapshotStamp()
{
    static std::atomic<uint64_t> stamp{0};
    return ++stamp;
}

namespace
{
double SecondsSince(std::chrono::steady_clock::time_point start) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }

namespace fs = std::filesystem;
using StatePart = AdaptiveDweaveLevelSet3D::StatePart;
constexpr int kCacheFormat = 8;  // bump when what init() computes or the parts change (4: level set .vdb; 8: springls cereal again)

// Names the initial state: the source mesh (path, size and modification time) and every setting init() reads.
std::string CacheTag(const MeshOptions &options, const MeshGridFit &fit)
{
    std::error_code ec;
    const fs::path mesh = fs::absolute(options.mesh, ec);
    std::ostringstream s;
    s << std::setprecision(17) << "springls-viewer cache " << kCacheFormat << "; mesh " << mesh.generic_string() << "; bytes "
      << fs::file_size(options.mesh, ec) << "; modified " << fs::last_write_time(options.mesh, ec).time_since_epoch().count() << "; cellScale "
      << options.grid.cellScale << "; grid " << options.grid.gridOverride << "; maxDepth " << options.grid.maxDepth << "; maxFinestCells "
      << options.grid.maxFinestCells << "; depth " << fit.maxDepth << "; baseCells " << fit.baseCells.x() << " " << fit.baseCells.y() << " "
      << fit.baseCells.z() << "; finestCell " << fit.finestCell << "; element " << (options.elementType == ElementType::Quad ? "quad" : "triangle")
      << "; settle " << options.settleIterations;
    for (const auto &[key, value] : options.tuning) s << "; " << key << " " << value;
    return s.str();
}

// cacheDir/<mesh stem>-<hash of the tag>.<part>.<vdb or cereal>
fs::path CachePath(const MeshOptions &options, const std::string &tag, StatePart part)
{
    uint64_t hash = 14695981039346656037ull;  // FNV-1a
    for (unsigned char ch : tag) hash = (hash ^ ch) * 1099511628211ull;
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
    return options.cacheDir / (options.mesh.stem().string() + "-" + hex + "." + AdaptiveDweaveLevelSet3D::StatePartName(part) + "." +
                               AdaptiveDweaveLevelSet3D::StatePartExtension(part));
}

// Restores sim from the cache; false (with a reason in error, empty when there is simply no cache) otherwise.
bool RestoreFromCache(AdaptiveDweaveLevelSet3D &sim, const MeshOptions &options, const std::string &tag, std::string &error)
{
    std::array<std::ifstream, AdaptiveDweaveLevelSet3D::kStatePartCount> in;
    std::array<std::istream *, AdaptiveDweaveLevelSet3D::kStatePartCount> streams{};
    for (size_t i = 0; i < in.size(); i++) {
        in[i].open(CachePath(options, tag, AdaptiveDweaveLevelSet3D::kStateParts[i]), std::ios::binary);
        if (!in[i]) return false;
        streams[i] = &in[i];
    }
    return sim.loadParts(streams, tag, error);
}

// Writes each part to a temporary file renamed into place, so an interrupted write leaves no partial part.
bool SaveToCache(const AdaptiveDweaveLevelSet3D &sim, const MeshOptions &options, const std::string &tag, std::string &error)
{
    std::error_code ec;
    fs::create_directories(options.cacheDir, ec);
    for (StatePart part : AdaptiveDweaveLevelSet3D::kStateParts) {
        const fs::path path = CachePath(options, tag, part);
        fs::path temporary = path;
        temporary += ".tmp";
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            if (!out) {
                error = "could not write " + temporary.string();
                return false;
            }
            if (!sim.savePart(part, out, tag, error)) return false;
        }
        fs::rename(temporary, path, ec);
        if (ec) {
            error = "could not write " + path.string() + ": " + ec.message();
            fs::remove(temporary, ec);
            return false;
        }
    }
    return true;
}
}  // namespace

MeshSimulation::MeshSimulation() : mEntry(*Registry::Instance().findKey(Domain::D3, Scenario::Mesh, "dweave")) {}

MeshSimulation::MeshSimulation(const Entry &entry, const MeshOptions &options) : mEntry(entry), mOptions(options) {}

bool MeshSimulation::reload(const MeshOptions &options, std::string &error) { return load(options, error); }

bool MeshSimulation::load(const MeshOptions &options, std::string &error)
{
    error.clear();
    // Everything is built into locals and swapped in at the end: a failed load keeps the previous session.
    PolygonMesh polygons;
    if (!LoadPly(options.mesh, polygons, error)) {
        if (error.empty()) error = "could not read " + options.mesh.string();
        return false;
    }
    const std::vector<std::array<uint32_t, 3>> triangles = Triangulate(polygons);
    if (polygons.positions.empty() || triangles.empty()) {
        error = "mesh has no faces: " + options.mesh.string();
        return false;
    }
    const auto start = std::chrono::steady_clock::now();

    MeshStats stats;
    stats.inputFaces = polygons.faceCount();
    stats.topology = AnalyzeTopology(polygons);
    std::vector<std::string> warnings;
    if (stats.topology.boundaryEdges > 0) {
        std::ostringstream w;
        w << "mesh is open: " << stats.topology.boundaryEdges << " boundary edges; the inside is ambiguous near them";
        warnings.push_back(w.str());
    }
    if (stats.topology.nonManifoldEdges > 0) {
        std::ostringstream w;
        w << "mesh has " << stats.topology.nonManifoldEdges << " non-manifold edges";
        warnings.push_back(w.str());
    }

    // Tuning first: it fails fast on a bad key, and the padding follows --set bandWidth.
    auto sim = std::make_unique<AdaptiveDweaveLevelSet3D>(options.elementType);
    for (const auto &[key, value] : options.tuning) {
        if (!SetTuning(*sim, key, value)) {
            error = "unknown tuning key '" + key + "'";
            return false;
        }
    }
    MeshGridOptions gridOptions = options.grid;
    gridOptions.padVoxels = 2.0f + sim->adaptiveTuning.bandWidth;
    const MeshGridFit fit = FitMeshGrid(polygons, gridOptions);
    if (fit.clamped) {
        std::ostringstream w;
        w << "resolution limited: finest cell is " << fit.finestCell << ", coarser than the mesh edge length asks for (cap of " << gridOptions.maxFinestCells
          << " finest cells along the longest axis)";
        warnings.push_back(w.str());
    }

    // Grid-space copy of the mesh: the distance object for the exact initial level set, and the input layer.
    Points3 gridPositions(polygons.positions.size());
    for (size_t i = 0; i < gridPositions.size(); i++) gridPositions[i] = fit.toGrid(polygons.positions[i]);

    sim->adaptiveTuning.maxDepth = fit.maxDepth;
    sim->setInitialLevelSet(VolumeF(fit.baseCells.x() + 1, fit.baseCells.y() + 1, fit.baseCells.z() + 1, sim->adaptiveTuning.bandWidth));
    const std::string tag = options.cacheDir.empty() ? std::string() : CacheTag(options, fit);
    if (!options.cacheDir.empty()) {
        std::string cacheError;
        stats.fromCache = RestoreFromCache(*sim, options, tag, cacheError);
        if (!cacheError.empty()) warnings.push_back("cache not used (" + cacheError + "); initializing");
    }
    if (!stats.fromCache) {
        auto distance = std::make_shared<MeshDistance3D>();
        if (!distance->build(gridPositions, triangles)) {
            error = "mesh has no usable triangles";
            return false;
        }
        sim->setInitialSurface(distance);
        if (!sim->init()) {
            error = "initialization failed";
            return false;
        }
        sim->setInitialSurface(nullptr);  // only init() reads it
        stats.settleIterations = sim->settle(options.settleIterations);
        std::string cacheError;
        if (!options.cacheDir.empty() && !SaveToCache(*sim, options, tag, cacheError)) warnings.push_back("cache not written: " + cacheError);
    }

    SurfaceMesh3D input;
    input.type = ElementType::Triangle;
    input.vertexes = std::move(gridPositions);
    input.indexes.reserve(triangles.size() * 3);
    for (const auto &t : triangles) input.indexes.insert(input.indexes.end(), t.begin(), t.end());
    input.updateVertexNormals();

    mInputMesh = std::move(input);
    mFit = fit;
    mSim = std::move(sim);
    mWarnings = std::move(warnings);
    stats.finestDepth = fit.maxDepth;
    stats.finestCell = fit.finestCell;
    stats.baseCells = fit.baseCells;
    stats.clamped = fit.clamped;
    stats.threads = MaxThreads();
    mStats = stats;
    updateStats();
    mStats.initSeconds = SecondsSince(start);
    mStamp = NextSnapshotStamp();
    mOptions = options;
    return true;
}

bool MeshSimulation::stepOnce()
{
    if (!mSim) return false;
    const auto start = std::chrono::steady_clock::now();
    mSim->step();
    mStats.steps++;
    updateStats();
    mStats.lastStepSeconds = SecondsSince(start);
    mStamp = NextSnapshotStamp();
    return true;
}

void MeshSimulation::setResampling(bool on)
{
    if (mSim) mSim->setResamplingEnabled(on);
}

void MeshSimulation::updateStats()
{
    mStats.primal = mSim->countSpringls(Constellation::Primal);
    mStats.dual = mSim->countSpringls(Constellation::Dual);
    mStats.isoVertexes = mSim->getManifold().isoSurface.vertexes.size();
    mStats.isoFaces = mSim->getManifold().isoSurface.faceCount();
    mStats.leaves = mSim->getGrid().leaves().size();
}

MeshSnapshot MeshSimulation::snapshot() const
{
    MeshSnapshot snap;
    snap.stamp = mStamp;
    snap.fit = mFit;
    snap.stats = mStats;
    snap.warnings = mWarnings;
    if (mSim) {
        snap.grid = std::make_shared<const AdaptiveGrid3D>(mSim->getGrid());
        snap.manifold = mSim->getManifold();
        snap.inputMesh = mInputMesh;
    }
    return snap;
}

void MeshSimulation::printInit(std::ostream &out) const
{
    const MeshSnapshot snap = snapshot();
    const MeshStats &s = snap.stats;
    const bool autoDepth = mOptions.grid.maxDepth < 0;
    Vec3 lo = Vec3::Zero(), hi = Vec3::Zero();
    if (!snap.inputMesh.vertexes.empty()) {
        lo = hi = snap.inputMesh.vertexes[0];
        for (const Vec3 &p : snap.inputMesh.vertexes) {
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
    }
    // The mesh's longest side in finest cells (the grid is larger by its padding).
    const long meshCells = std::lround((hi - lo).maxCoeff() * std::ldexp(1.0, s.finestDepth));
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "depth " << s.finestDepth << (autoDepth ? " (auto)" : "") << ", finest cell " << s.finestCell << ", base cells " << s.baseCells.x() << "x"
        << s.baseCells.y() << "x" << s.baseCells.z() << ", finest cells across the mesh " << meshCells << " (grid "
        << (snap.fit.baseCells.maxCoeff() << s.finestDepth) << "), " << s.inputFaces << " input faces (" << snap.inputMesh.faceCount() << " triangles), init "
        << std::fixed << std::setprecision(2) << s.initSeconds << " s" << (s.fromCache ? " (restored from the cache)" : "");
    if (!s.fromCache) out << ", settled in " << s.settleIterations << " iterations";
    out << std::endl;
    out.flags(flags);
    out.precision(precision);
}

void MeshSimulation::printStep(std::ostream &out) const
{
    const MeshStats &s = mStats;
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << "step " << s.steps << ": primal " << s.primal << ", dual " << s.dual << ", iso faces " << s.isoFaces << ", leaves " << s.leaves << ", " << std::fixed
        << std::setprecision(3) << s.lastStepSeconds << " s" << std::endl;
    out.flags(flags);
    out.precision(precision);
}
}  // namespace toybox
