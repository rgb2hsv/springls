// Mesh tracking (MeshTracking3D): a closed mesh advected with the flow and kept well shaped by adaptive remeshing.
#include "Check.hpp"

#include <lod/MeshTracking3D.hpp>
#include <scenarios/EnrightTest3D.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <numbers>

using namespace springls;

namespace
{
bool Closed(const TriangleSurface &mesh)
{
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (const auto &t : mesh.triangles)
        for (int e = 0; e < 3; e++) edges[std::minmax(t[e], t[(e + 1) % 3])]++;
    for (const auto &[edge, n] : edges)
        if (n != 2) return false;
    return !mesh.triangles.empty();
}
}  // namespace

// Through the registry, on a short Enright run on a 64^3 grid (128^3 equivalent at depth 1): the mesh stays closed,
// is remeshed as it stretches, and the level set rebuilt from it keeps the volume.
TEST(mesh_tracking_enright)
{
    EnrightConfig3D config;
    config.gridSize = 64;
    config.period = 0.5;
    auto sim = MakeSimulation(Method3D::MeshTracking, ElementType::Triangle, 1);
    CHECK(sim && SetTuning(*sim, "sphereLevels", 4));
    if (!sim) return;
    ConfigureEnright(*sim, config);
    CHECK(sim->init());
    const auto *tracking = dynamic_cast<const MeshTracking3D *>(sim.get());
    CHECK(tracking != nullptr);
    if (!tracking) return;
    const size_t start = tracking->triangleCount();
    size_t most = start;
    int steps = 0;
    while (sim->step() && steps < 100000) {
        steps++;
        most = std::max(most, tracking->triangleCount());
    }
    const EnrightMetrics3D metrics = MeasureEnright(*sim, config);
    // The mesh's own enclosed volume against the sphere's.
    double meshVolume = 0.0;
    for (const auto &t : tracking->mesh().triangles) {
        const Vec3 &p0 = tracking->mesh().vertexes[t[0]], &p1 = tracking->mesh().vertexes[t[1]], &p2 = tracking->mesh().vertexes[t[2]];
        meshVolume += p0.dot(p1.cross(p2)) / 6.0;
    }
    const double r = config.radiusVoxels();
    const double meshVolumeError = std::abs(meshVolume) / (4.0 / 3.0 * std::numbers::pi * r * r * r) - 1.0;
    std::printf("  %d steps, %zu -> at most %zu -> %zu triangles, volume error: level set %.4f, mesh %.4f%c", steps, start, most, tracking->triangleCount(),
                metrics.relativeVolumeError, meshVolumeError, 10);
    CHECK(Closed(tracking->mesh()));
    CHECK(most > start);  // the stretched mesh gained triangles
    // The level set is rebuilt from the mesh each sub-step (semi-Lagrangian): both keep the volume.
    CHECK(std::isfinite(metrics.relativeVolumeError) && std::abs(metrics.relativeVolumeError) < 0.02);
    CHECK(std::isfinite(meshVolumeError) && std::abs(meshVolumeError) < 0.01);
    CHECK(std::string(MethodKey(Method3D::MeshTracking)) == "mesh-tracking" && IsAdaptive(Method3D::MeshTracking));
}
