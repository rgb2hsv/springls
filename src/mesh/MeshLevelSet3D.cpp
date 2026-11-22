#include <mesh/MeshLevelSet3D.hpp>

#include <spatial/AdaptiveGrid3D.hpp>

#include <algorithm>
#include <cmath>

namespace springls
{
namespace
{
constexpr int kBlock = AdaptiveGrid3D::kBlockCells;

int RoundUpToBlock(int n) { return std::max(kBlock, (n + kBlock - 1) / kBlock * kBlock); }

Vec3i CellsFor(const Vec3 &extent, float baseVoxel, float pad)
{
    Vec3i cells;
    for (int a = 0; a < 3; a++) {
        // The epsilon keeps a quotient that is an exact integer up to float rounding from rounding up.
        const float raw = std::min(extent[a] / baseVoxel + 2.0f * pad - 1e-4f, 1e6f);
        cells[a] = RoundUpToBlock(static_cast<int>(std::ceil(raw)));
    }
    return cells;
}
}  // namespace

int ChooseMaxDepth(size_t triangleCount)
{
    if (triangleCount == 0) return 1;
    const double depth = std::round(std::log2(std::sqrt(static_cast<double>(triangleCount)) / 24.0));
    return static_cast<int>(std::clamp(depth, 1.0, static_cast<double>(AdaptiveGrid3D::kMaxSupportedDepth)));
}

MeshGridFit FitMeshGrid(const PolygonMesh &mesh, const MeshGridOptions &options)
{
    MeshGridFit fit;
    if (options.maxDepth < 0) {
        size_t triangles = 0;
        for (size_t f = 0; f < mesh.faceCount(); f++) {
            const size_t size = mesh.faceStart[f + 1] - mesh.faceStart[f];
            if (size >= 3) triangles += size - 2;
        }
        fit.maxDepth = ChooseMaxDepth(triangles);
    } else {
        fit.maxDepth = std::min(options.maxDepth, AdaptiveGrid3D::kMaxSupportedDepth);
    }
    const float pad = std::max(0.0f, options.padVoxels);
    const float depthScale = static_cast<float>(1 << fit.maxDepth);

    Vec3 lo = Vec3::Zero(), hi = Vec3::Zero();
    if (!mesh.positions.empty()) {
        lo = hi = mesh.positions[0];
        for (const Vec3 &p : mesh.positions) {
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
    }
    const Vec3 extent = hi - lo;
    const float longest = extent.maxCoeff() > 0.0f ? extent.maxCoeff() : 1.0f;

    float baseVoxel;
    if (options.gridOverride > 0) {
        // The longest axis gets exactly gridOverride cells (rounded up to a block), the mesh plus
        // padding spanning it.
        const int minCells = static_cast<int>(std::floor(2.0f * pad)) + 1;
        const int target = RoundUpToBlock(std::max(options.gridOverride, minCells + 1));
        baseVoxel = longest / (static_cast<float>(target) - 2.0f * pad);
    } else {
        float finest = MeanEdgeLength(mesh) * options.cellScale;
        if (!(finest > 0.0f) || !std::isfinite(finest)) finest = longest / 64.0f;
        baseVoxel = finest * depthScale;
    }

    // Cap the finest resolution along the longest axis by growing the base voxel.
    Vec3i cells = CellsFor(extent, baseVoxel, pad);
    // At least what the padding alone needs, so the cap is reachable at every depth.
    const int minBaseCells = RoundUpToBlock(static_cast<int>(std::ceil(2.0f * pad)) + 1);
    const int maxBaseCells = std::max(minBaseCells, (options.maxFinestCells >> fit.maxDepth) / kBlock * kBlock);
    if (cells.maxCoeff() > maxBaseCells) {
        fit.clamped = true;
        const float usable = std::max(1.0f, static_cast<float>(maxBaseCells) - 2.0f * pad);
        baseVoxel = std::max(baseVoxel, longest / usable);
        cells = CellsFor(extent, baseVoxel, pad);
        for (int i = 0; i < 256 && cells.maxCoeff() > maxBaseCells; i++) {
            baseVoxel *= 1.02f;
            cells = CellsFor(extent, baseVoxel, pad);
        }
    }

    fit.baseVoxel = baseVoxel;
    fit.finestCell = baseVoxel / depthScale;
    fit.baseCells = cells;
    fit.origin = 0.5f * (lo + hi) - 0.5f * baseVoxel * cells.cast<float>();
    return fit;
}
}  // namespace springls
