#include "Check.hpp"

#include <core/Enright2DSimulation.hpp>
#include <core/Enright3DSimulation.hpp>
#include <core/MeshSimulation.hpp>
#include <core/Registry.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace toybox;

namespace
{
const Entry &Find(Domain domain, Scenario scenario, const char *key)
{
    const Entry *e = Registry::Instance().findKey(domain, scenario, key);
    if (!e) std::abort();
    return *e;
}

std::unique_ptr<Simulation> Make(const Entry &entry, const RunOptions &options)
{
    std::unique_ptr<Simulation> sim = entry.make(options);
    std::string error;
    if (!sim || !sim->init(error)) {
        std::printf("init of %s failed: %s\n", entry.key.c_str(), error.c_str());
        return nullptr;
    }
    return sim;
}

// A closed cube [0, size]^3 of 1-unit quads, as ASCII PLY.
std::filesystem::path WriteCube(const std::string &name, int size)
{
    std::ostringstream verts, faces;
    int vertexCount = 0, faceCount = 0;
    for (int face = 0; face < 6; face++) {
        const int axis = face / 2, side = face % 2;
        for (int a = 0; a < size; a++) {
            for (int b = 0; b < size; b++) {
                int ids[4];
                const int corners[4][2] = {{a, b}, {a + 1, b}, {a + 1, b + 1}, {a, b + 1}};
                for (int k = 0; k < 4; k++) {
                    int p[3];
                    p[axis] = side * size;
                    p[(axis + 1) % 3] = corners[k][0];
                    p[(axis + 2) % 3] = corners[k][1];
                    verts << p[0] << " " << p[1] << " " << p[2] << "\n";
                    ids[k] = vertexCount++;
                }
                faces << "4";
                for (int k = 0; k < 4; k++) faces << " " << ids[side == 1 ? k : 3 - k];
                faces << "\n";
                faceCount++;
            }
        }
    }
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out << "ply\nformat ascii 1.0\nelement vertex " << vertexCount << "\nproperty float x\nproperty float y\nproperty float z\n"
        << "element face " << faceCount << "\nproperty list uchar int vertex_indices\nend_header\n"
        << verts.str() << faces.str();
    return path;
}
}  // namespace

TEST(adapter_2d_csv_header_matches_today)
{
    RunOptions options;
    options.gridSize = 32;
    options.period = 0.25;
    auto sim = Find(Domain::D2, Scenario::Enright, "springls").make(options);
    std::ostringstream header;
    sim->csvHeader(header);
    CHECK(header.str() ==
          "method,step,time,area,relative_area_error,mean_shape_error,max_shape_error,springls,springls_primal,springls_dual,step_seconds,substeps\n");
    CHECK(sim->supportsCsv() && sim->supportsCheckpoints());
}

TEST(adapter_2d_runs_enright_small)
{
    RunOptions options;
    options.gridSize = 32;
    options.period = 0.25;
    auto sim = Make(Find(Domain::D2, Scenario::Enright, "springls"), options);
    CHECK(sim != nullptr);
    if (!sim) return;
    auto *adapter = dynamic_cast<Enright2DSimulation *>(sim.get());
    CHECK(adapter != nullptr && adapter->config().gridSize == 32 && adapter->method() == springls::Method::Springls);
    CHECK(sim->time() == 0.0 && sim->springlCount() > 0 && !sim->complete());
    int steps = 0;
    while (sim->step() && steps < 100000) steps++;
    CHECK(sim->complete());
    CHECK_NEAR(sim->time(), 0.25, 1e-9);
    CHECK(sim->iteration() > 0 && sim->substeps() > 0);
    std::ostringstream row;
    sim->csvRow(row, 0.5);
    CHECK(row.str().rfind("springls," + std::to_string(sim->iteration()) + ",0.25,", 0) == 0);
}

TEST(adapter_2d_labels_rows_for_multi)
{
    RunOptions options;
    options.gridSize = 32;
    options.period = 0.25;
    auto sim = Make(Find(Domain::D2, Scenario::Enright, "multi-levelset"), options);
    CHECK(sim != nullptr && sim->hasLabelsCsv());
    if (!sim) return;
    std::ostringstream header, rows;
    sim->labelsCsvHeader(header);
    CHECK(header.str() == "method,step,time,label,area,relative_area_error\n");
    sim->labelsCsvRows(rows);
    const std::string s = rows.str();
    CHECK(s.rfind("multi-levelset,0,0,1,", 0) == 0);
    CHECK(s.find("multi-levelset,0,0,4,") != std::string::npos);
    auto single = Make(Find(Domain::D2, Scenario::Enright, "dweave"), options);
    CHECK(single && !single->hasLabelsCsv());
}

TEST(adapter_3d_csv_header_matches_today)
{
    RunOptions options;
    options.gridSize = 16;
    auto sim = Find(Domain::D3, Scenario::Enright, "dweave").make(options);
    std::ostringstream header;
    sim->csvHeader(header);
    CHECK(header.str() ==
          "method,elements,grid,step,time,volume,relative_volume_error,mean_shape_error,max_shape_error,springls,springls_primal,springls_dual,step_seconds,"
          "substeps\n");
    CHECK(sim->supportsCsv() && sim->supportsCheckpoints());
}

TEST(adapter_3d_runs_small)
{
    RunOptions options;
    options.gridSize = 16;
    options.period = 0.5;
    options.maxDepth = 2;
    auto sim = Make(Find(Domain::D3, Scenario::Enright, "dweave"), options);
    CHECK(sim != nullptr);
    if (!sim) return;
    for (int i = 0; i < 5; i++) CHECK(sim->step());
    CHECK(sim->iteration() == 5 && sim->time() > 0.0 && !sim->complete());
    auto *adapter = dynamic_cast<Enright3DSimulation *>(sim.get());
    CHECK(adapter != nullptr);
    if (!adapter) return;
    const RunSnapshot snap = adapter->snapshot();
    CHECK(snap.iteration == 5 && snap.config.gridSize == 16 && snap.levelSet != nullptr && snap.grid == nullptr);
    std::ostringstream row;
    sim->csvRow(row, 0.0);
    CHECK(row.str().rfind("dweave,quad,16,5,", 0) == 0);
}

TEST(adapter_3d_checkpoint_round_trip)
{
    RunOptions options;
    options.gridSize = 16;
    options.period = 0.5;
    options.maxDepth = 2;
    const Entry &entry = Find(Domain::D3, Scenario::Enright, "adaptive-dweave");
    auto sim = Make(entry, options);
    CHECK(sim != nullptr);
    if (!sim) return;
    for (int i = 0; i < 3; i++) sim->step();
    const auto dir = std::filesystem::temp_directory_path() / "toybox_checkpoint_round_trip";
    std::filesystem::remove_all(dir);
    std::string error;
    CHECK(sim->saveCheckpoint(dir, 0.01, error));
    const auto state = dir / "adaptive-dweave_quad_g16_d2_t0.0100.state";
    CHECK(std::filesystem::exists(state) && std::filesystem::exists(dir / "adaptive-dweave_quad_g16_d2_t0.0100.vdb"));
    auto resumed = Make(entry, options);
    CHECK(resumed != nullptr);
    if (!resumed) return;
    CHECK(resumed->resume(state, error));
    CHECK(resumed->iteration() == sim->iteration() && resumed->time() == sim->time() && resumed->springlCount() == sim->springlCount());
    CHECK(!resumed->resume(dir / "missing.state", error) && !error.empty());
}

TEST(mesh_simulation_reload_uses_cache)
{
    const auto dir = std::filesystem::temp_directory_path() / "toybox_mesh_cache";
    std::filesystem::remove_all(dir);
    RunOptions options;
    options.mesh.mesh = WriteCube("toybox_mesh_cache.ply", 16);
    options.mesh.grid.maxDepth = 2;
    options.mesh.cacheDir = dir;
    const Entry &entry = Find(Domain::D3, Scenario::Mesh, "dweave");
    auto first = Make(entry, options);
    CHECK(first != nullptr);
    if (!first) return;
    auto *mesh = dynamic_cast<MeshSimulation *>(first.get());
    CHECK(mesh != nullptr && !mesh->snapshot().stats.fromCache);
    CHECK(!first->supportsCsv() && !first->complete());
    // A rebuild through the factory with the same options restores the cached state.
    auto second = Make(entry, options);
    CHECK(second && dynamic_cast<MeshSimulation *>(second.get())->snapshot().stats.fromCache);
    // reload() in place with the same options also hits the cache.
    std::string error;
    CHECK(mesh->reload(options.mesh, error) && mesh->snapshot().stats.fromCache);
    CHECK(first->step() && first->iteration() == 1 && first->springlCount() > 0);
}

TEST(registry_every_entry_constructs_and_inits)
{
    const auto cube = WriteCube("toybox_every_entry.ply", 12);
    for (const Entry &entry : Registry::Instance().entries()) {
        if (entry.config.scenario == Scenario::Sphere) continue;  // factory arrives with SphereSimulation (next task)
        RunOptions options;
        options.gridSize = 16;
        options.period = 0.5;
        options.maxDepth = 1;
        options.mesh.mesh = cube;
        options.mesh.grid.maxDepth = 1;
        auto sim = Make(entry, options);
        CHECK(sim != nullptr);
        if (!sim) continue;
        CHECK(&sim->entry() == &entry);
        CHECK(sim->step());
    }
}

// A window rebuilds its mesh simulation through the factory on Reset and edits: the new one's snapshots
// must not reuse the old one's stamps, or the views keep drawing the old surface.
TEST(mesh_simulation_stamps_continue_across_rebuilds)
{
    RunOptions options;
    options.mesh.mesh = WriteCube("toybox_mesh_stamps.ply", 8);
    options.mesh.grid.maxDepth = 1;
    const Entry &entry = Find(Domain::D3, Scenario::Mesh, "dweave");
    auto first = Make(entry, options);
    auto second = Make(entry, options);
    CHECK(first && second);
    if (!first || !second) return;
    const uint64_t a = dynamic_cast<MeshSimulation *>(first.get())->snapshot().stamp;
    const uint64_t b = dynamic_cast<MeshSimulation *>(second.get())->snapshot().stamp;
    CHECK(b > a);
    first->step();
    CHECK(dynamic_cast<MeshSimulation *>(first.get())->snapshot().stamp > b);
}
