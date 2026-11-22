#include "Check.hpp"

#include <io/Ply.hpp>
#include <mesh/MeshLevelSet3D.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

using namespace springls;

namespace
{
// Closed cube [0, size]^3 with n x n outward-wound quads per face (edge length size / n).
PolygonMesh MakeGridCube(float size, int n)
{
    PolygonMesh mesh;
    const int stride = n + 1;
    auto id = [&](int face, int a, int b) { return static_cast<uint32_t>(face * stride * stride + a * stride + b); };
    // Each face has its own vertexes; axis is the fixed axis, side is 0 or 1.
    for (int face = 0; face < 6; face++) {
        const int axis = face / 2, side = face % 2;
        const int u = (axis + 1) % 3, v = (axis + 2) % 3;
        for (int a = 0; a <= n; a++) {
            for (int b = 0; b <= n; b++) {
                Vec3 p = Vec3::Zero();
                p[axis] = side * size;
                p[u] = size * a / n;
                p[v] = size * b / n;
                mesh.positions.push_back(p);
            }
        }
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                // (u, v) is counterclockwise seen from +axis, so flip for the low side.
                const uint32_t q[4] = {id(face, a, b), id(face, a + 1, b), id(face, a + 1, b + 1), id(face, a, b + 1)};
                if (side == 1) {
                    for (int k = 0; k < 4; k++) mesh.faceVertexes.push_back(q[k]);
                } else {
                    for (int k = 3; k >= 0; k--) mesh.faceVertexes.push_back(q[k]);
                }
                mesh.faceStart.push_back(static_cast<uint32_t>(mesh.faceVertexes.size()));
            }
        }
    }
    return mesh;
}

Vec3 BoundsCenter(const PolygonMesh &mesh)
{
    Vec3 lo = mesh.positions[0], hi = mesh.positions[0];
    for (const Vec3 &p : mesh.positions) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    return 0.5f * (lo + hi);
}
}  // namespace

TEST(fit_from_edge_length)
{
    const PolygonMesh mesh = MakeGridCube(10.0f, 10);
    MeshGridOptions options;
    options.maxDepth = 3;
    const MeshGridFit fit = FitMeshGrid(mesh, options);
    CHECK_NEAR(fit.finestCell, 1.0f, 1e-4);
    CHECK_NEAR(fit.baseVoxel, 8.0f, 1e-3);
    CHECK(fit.maxDepth == 3);
    CHECK(!fit.clamped);
    for (int a = 0; a < 3; a++) {
        CHECK(fit.baseCells[a] % 8 == 0);
        CHECK(fit.baseCells[a] >= 12);
    }
    const Vec3 gridCenter = 0.5f * fit.baseCells.cast<float>();
    const Vec3 g = fit.toGrid(BoundsCenter(mesh));
    for (int a = 0; a < 3; a++) CHECK_NEAR(g[a], gridCenter[a], 1e-4);
    const Vec3 back = fit.toMesh(g);
    for (int a = 0; a < 3; a++) CHECK_NEAR(back[a], BoundsCenter(mesh)[a], 1e-4);
}

TEST(fit_cell_scale_and_override)
{
    const PolygonMesh mesh = MakeGridCube(10.0f, 10);
    MeshGridOptions options;
    options.maxDepth = 3;
    options.cellScale = 2.0f;
    CHECK_NEAR(FitMeshGrid(mesh, options).finestCell, 2.0f, 1e-4);

    options.cellScale = 1.0f;
    options.gridOverride = 32;
    const MeshGridFit fit = FitMeshGrid(mesh, options);
    CHECK(fit.baseCells.maxCoeff() == 32);
    CHECK_NEAR(fit.baseVoxel, 10.0f / (32.0f - 10.0f), 1e-4);
    CHECK_NEAR(fit.finestCell, fit.baseVoxel / 8.0f, 1e-5);

    // Too small an override is raised so the padded mesh still fits.
    options.gridOverride = 4;
    const MeshGridFit small = FitMeshGrid(mesh, options);
    CHECK(small.baseCells.maxCoeff() == 16);
    CHECK(small.baseVoxel > 0.0f);
}

TEST(fit_clamps_tiny_triangles)
{
    const PolygonMesh mesh = MakeGridCube(10.0f, 10);
    MeshGridOptions options;
    options.cellScale = 1e-5f;
    const MeshGridFit fit = FitMeshGrid(mesh, options);
    CHECK(fit.clamped);
    CHECK((fit.baseCells.maxCoeff() << fit.maxDepth) <= options.maxFinestCells + 8 * (1 << fit.maxDepth));
    CHECK_NEAR(fit.finestCell, fit.baseVoxel / static_cast<float>(1 << fit.maxDepth), 1e-5);
}

TEST(fit_cap_holds_at_every_depth)
{
    const PolygonMesh mesh = MakeGridCube(10.0f, 10);
    for (int depth = 0; depth <= 12; depth++) {
        MeshGridOptions options;
        options.maxDepth = depth;
        options.cellScale = 1e-5f;
        const MeshGridFit fit = FitMeshGrid(mesh, options);
        CHECK(fit.maxDepth == std::min(depth, 10));
        CHECK(fit.clamped);
        CHECK((fit.baseCells.maxCoeff() << fit.maxDepth) <= options.maxFinestCells + 8 * (1 << fit.maxDepth));
    }
}

TEST(choose_depth_from_triangles)
{
    CHECK(ChooseMaxDepth(0) == 1);
    CHECK(ChooseMaxDepth(1000) == 1);
    CHECK(ChooseMaxDepth(10000) == 2);
    CHECK(ChooseMaxDepth(132524) == 4);
    CHECK(ChooseMaxDepth(599642) == 5);
    CHECK(ChooseMaxDepth(4000000) == 6);
    CHECK(ChooseMaxDepth(16000000) == 7);
    CHECK(ChooseMaxDepth(1000000000000ull) == 10);
    CHECK(FitMeshGrid(MakeGridCube(2.0f, 1), MeshGridOptions()).maxDepth == 1);  // 12 triangles
}

TEST(fit_survives_degenerate_mesh)
{
    const MeshGridFit empty = FitMeshGrid(PolygonMesh(), MeshGridOptions());
    CHECK(std::isfinite(empty.baseVoxel) && empty.baseVoxel > 0.0f);
    CHECK(empty.baseCells.minCoeff() >= 8);
}

TEST(fit_override_is_exact_and_not_clamped)
{
    const PolygonMesh mesh = MakeGridCube(10.0f, 10);
    for (int n : {32, 40, 64}) {
        MeshGridOptions options;
        options.gridOverride = n;
        const MeshGridFit fit = FitMeshGrid(mesh, options);
        CHECK(!fit.clamped);
        CHECK(fit.baseCells.maxCoeff() == n);
    }
}
