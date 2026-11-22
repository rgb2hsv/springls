#include "Check.hpp"

#include <core/MeshSimulation.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace springls;

namespace
{
// ASCII PLY of the cube [0, size]^3 with n x n outward-wound quads per face; skipFace (0..5) leaves a face out.
std::filesystem::path WriteCubePly(const std::string &name, int size, int skipFace)
{
    const int n = size;  // 1-unit quads
    // Vertexes are shared between faces (welded by integer position).
    std::map<std::array<int, 3>, int> ids;
    std::ostringstream verts, faces;
    auto vertex = [&](int face, int a, int b) {
        const int axis = face / 2, side = face % 2;
        std::array<int, 3> p;
        p[axis] = side * size;
        p[(axis + 1) % 3] = a;
        p[(axis + 2) % 3] = b;
        auto it = ids.find(p);
        if (it != ids.end()) return it->second;
        const int id = static_cast<int>(ids.size());
        ids[p] = id;
        verts << p[0] << " " << p[1] << " " << p[2] << "\n";
        return id;
    };
    int faceCount = 0;
    for (int face = 0; face < 6; face++) {
        if (face == skipFace) continue;
        const int side = face % 2;
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                const int q[4] = {vertex(face, a, b), vertex(face, a + 1, b), vertex(face, a + 1, b + 1), vertex(face, a, b + 1)};
                faces << "4";
                for (int k = 0; k < 4; k++) faces << " " << q[(side == 1) ? k : 3 - k];
                faces << "\n";
                faceCount++;
            }
        }
    }
    // A skipped face leaves its interior vertexes unused; that is fine for the loader.
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out << "ply\nformat ascii 1.0\nelement vertex " << ids.size() << "\nproperty float x\nproperty float y\nproperty float z\n"
        << "element face " << faceCount << "\nproperty list uchar int vertex_indices\nend_header\n"
        << verts.str() << faces.str();
    return path;
}
}  // namespace

TEST(session_cube_runs)
{
    const auto path = WriteCubePly("springls_session_cube.ply", 20, -1);
    toybox::MeshOptions options;
    options.mesh = path;
    options.grid.maxDepth = 2;
    toybox::MeshSimulation session;
    std::string error;
    CHECK(session.load(options, error));
    CHECK(error.empty());
    for (int i = 0; i < 3; i++) CHECK(session.step());
    const toybox::MeshSnapshot snap = session.snapshot();
    CHECK(snap.stats.steps == 3);
    CHECK(snap.stats.primal > 0);
    CHECK(snap.stats.dual > 0);
    CHECK(snap.stats.isoFaces > 0);
    CHECK(snap.stats.leaves > 0);
    CHECK(snap.stats.finestDepth == 2);
    CHECK(snap.stats.settleIterations >= 1 && snap.stats.settleIterations <= 20);
    CHECK(snap.stats.inputFaces == 6 * 400);
    CHECK(snap.grid != nullptr);
    CHECK(snap.inputMesh.faceCount() == 6 * 400 * 2);
    CHECK(snap.stamp >= 4);
    CHECK(snap.warnings.empty());
    std::filesystem::remove(path);
}

TEST(session_open_mesh_warns)
{
    const auto path = WriteCubePly("springls_session_open.ply", 20, 5);
    toybox::MeshOptions options;
    options.mesh = path;
    options.grid.maxDepth = 2;
    toybox::MeshSimulation session;
    std::string error;
    CHECK(session.load(options, error));
    bool found = false;
    for (const std::string &w : session.snapshot().warnings) found |= w.find("boundary") != std::string::npos && w.find("80") != std::string::npos;
    CHECK(found);
    std::filesystem::remove(path);
}

TEST(session_settle_limits)
{
    const auto path = WriteCubePly("springls_session_settle.ply", 20, -1);
    toybox::MeshOptions options;
    options.mesh = path;
    options.grid.maxDepth = 2;
    std::string error;
    options.settleIterations = 0;
    toybox::MeshSimulation none;
    CHECK(none.load(options, error));
    CHECK(none.snapshot().stats.settleIterations == 0);
    options.settleIterations = 1;
    toybox::MeshSimulation one;
    CHECK(one.load(options, error));
    CHECK(one.snapshot().stats.settleIterations == 1);
    std::filesystem::remove(path);
}

TEST(session_bad_file)
{
    toybox::MeshOptions options;
    options.mesh = std::filesystem::temp_directory_path() / "springls_no_such_file.ply";
    toybox::MeshSimulation session;
    std::string error;
    CHECK(!session.load(options, error));
    CHECK(!error.empty());
}

TEST(session_failed_load_keeps_session)
{
    const auto path = WriteCubePly("springls_session_keep.ply", 20, -1);
    toybox::MeshOptions options;
    options.mesh = path;
    options.grid.maxDepth = 2;
    toybox::MeshSimulation session;
    std::string error;
    CHECK(session.load(options, error));
    CHECK(session.step());
    const toybox::MeshSnapshot before = session.snapshot();

    toybox::MeshOptions bad = options;
    bad.mesh = std::filesystem::temp_directory_path() / "springls_no_such_file.ply";
    CHECK(!session.load(bad, error));
    CHECK(!error.empty());
    bad = options;
    bad.tuning.emplace_back("noSuchKey", 1.0);
    CHECK(!session.load(bad, error));
    CHECK(!error.empty());

    CHECK(session.loaded());
    const toybox::MeshSnapshot after = session.snapshot();
    CHECK(after.stamp == before.stamp);
    CHECK(after.stats.steps == before.stats.steps);
    CHECK(after.stats.primal == before.stats.primal);
    CHECK(after.grid != nullptr);
    CHECK(session.step());
    std::filesystem::remove(path);
}

namespace
{
std::vector<std::filesystem::path> CacheFiles(const std::filesystem::path &dir)
{
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_directory(dir)) {
        for (const auto &entry : std::filesystem::directory_iterator(dir)) files.push_back(entry.path());
    }
    return files;
}

float MaxDifference(const Points3 &a, const Points3 &b)
{
    if (a.size() != b.size()) return 1e30f;
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); i++) worst = std::max(worst, (a[i] - b[i]).norm());
    return worst;
}
}  // namespace

TEST(session_cache_round_trip)
{
    const auto path = WriteCubePly("springls_session_cache.ply", 20, -1);
    const auto dir = std::filesystem::temp_directory_path() / "springls_session_cache_dir";
    std::filesystem::remove_all(dir);
    toybox::MeshOptions options;
    options.mesh = path;
    options.grid.maxDepth = 2;
    options.cacheDir = dir;
    std::string error;

    toybox::MeshSimulation cold;
    CHECK(cold.load(options, error));
    CHECK(!cold.snapshot().stats.fromCache);
    CHECK(cold.snapshot().warnings.empty());
    const auto files = CacheFiles(dir);
    CHECK(files.size() == 5);
    for (const auto &f : files) {
        const std::string name = f.filename().string();
        CHECK(name.rfind("springls_session_cache-", 0) == 0);
        const bool levelSet = name.find(".levelset.") != std::string::npos;
        const std::string extension = levelSet ? ".vdb" : ".cereal";
        CHECK(name.size() > extension.size() && name.substr(name.size() - extension.size()) == extension);
    }

    toybox::MeshSimulation warm;
    CHECK(warm.load(options, error));
    const toybox::MeshSnapshot a = cold.snapshot(), b = warm.snapshot();
    CHECK(b.stats.fromCache);
    CHECK(b.warnings.empty());
    CHECK(a.stats.primal == b.stats.primal);
    CHECK(a.stats.dual == b.stats.dual);
    CHECK(a.stats.leaves == b.stats.leaves);
    CHECK(a.manifold.springlParticles == b.manifold.springlParticles);
    CHECK(a.manifold.springlVertexes == b.manifold.springlVertexes);
    CHECK(a.manifold.springlStart == b.manifold.springlStart);
    CHECK(a.manifold.springlConstellations == b.manifold.springlConstellations);
    CHECK(a.manifold.isoSurface.vertexes == b.manifold.isoSurface.vertexes);
    CHECK(a.manifold.isoSurface.indexes == b.manifold.isoSurface.indexes);

    // The restored simulation steps like the original.
    for (int i = 0; i < 2; i++) {
        CHECK(cold.step());
        CHECK(warm.step());
    }
    const toybox::MeshSnapshot c = cold.snapshot(), d = warm.snapshot();
    CHECK(c.stats.primal == d.stats.primal);
    CHECK(c.stats.dual == d.stats.dual);
    CHECK(c.stats.isoFaces == d.stats.isoFaces);
    CHECK(MaxDifference(c.manifold.springlParticles, d.manifold.springlParticles) < 1e-4f);

    // Other settings are a different cache entry.
    toybox::MeshOptions coarser = options;
    coarser.grid.cellScale = 1.5f;
    toybox::MeshSimulation other;
    CHECK(other.load(coarser, error));
    CHECK(!other.snapshot().stats.fromCache);
    CHECK(CacheFiles(dir).size() == 10);

    std::filesystem::remove_all(dir);
    std::filesystem::remove(path);
}

TEST(session_cache_damaged_falls_back)
{
    const auto path = WriteCubePly("springls_session_damaged.ply", 20, -1);
    const auto dir = std::filesystem::temp_directory_path() / "springls_session_damaged_dir";
    std::filesystem::remove_all(dir);
    toybox::MeshOptions options;
    options.mesh = path;
    options.grid.maxDepth = 2;
    options.cacheDir = dir;
    std::string error;
    toybox::MeshSimulation first;
    CHECK(first.load(options, error));
    for (const auto &f : CacheFiles(dir)) {
        if (f.filename().string().find(".primal.") != std::string::npos) std::filesystem::resize_file(f, std::filesystem::file_size(f) / 2);
    }

    toybox::MeshSimulation second;
    CHECK(second.load(options, error));
    const toybox::MeshSnapshot s = second.snapshot();
    CHECK(!s.stats.fromCache);
    CHECK(s.stats.primal == first.snapshot().stats.primal);
    bool warned = false;
    for (const std::string &w : s.warnings) warned |= w.find("cache") != std::string::npos;
    CHECK(warned);

    // Initializing again rewrote the damaged part, so the next load restores.
    toybox::MeshSimulation restored;
    CHECK(restored.load(options, error));
    CHECK(restored.snapshot().stats.fromCache);

    // A damaged level set (.vdb) also falls back.
    for (const auto &f : CacheFiles(dir)) {
        if (f.filename().string().find(".levelset.vdb") != std::string::npos) std::filesystem::resize_file(f, std::filesystem::file_size(f) / 2);
    }
    toybox::MeshSimulation vdbDamaged;
    CHECK(vdbDamaged.load(options, error));
    CHECK(!vdbDamaged.snapshot().stats.fromCache);
    toybox::MeshSimulation third;
    CHECK(third.load(options, error));
    CHECK(third.snapshot().stats.fromCache);
    std::filesystem::remove_all(dir);
    std::filesystem::remove(path);
}
