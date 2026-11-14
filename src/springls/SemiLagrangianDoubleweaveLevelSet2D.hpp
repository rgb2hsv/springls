#pragma once
#include <springls/AdaptiveSpringLevelSet2D.hpp>

#include <limits>
#include <vector>

namespace springls
{
// Semi-Lagrangian doubleweave springls on the adaptive grid.
//
// Springls are advected freely, and every grid (the combined one and each constellation's) is
// rebuilt from its springls after each sub-step, before relaxation, with SpringlSignField2D: the
// min unsigned distance to the springls, signed by topology (an exterior flood around the
// springls' capsules and an upwind sweep into them), with no springl normals involved. The level
// set never moves on its own, so there is no level-set PDE and no CFL limit: the sub-step is
// limited by how much the flow stretches springls (strain), plus the advection schedule.
// Topology preservation is not supported. Compare AdaptiveDoubleweaveLevelSet2D, whose level set
// chases the springls with an upwind PDE and so moves them at most tuning.maxStepPixels per
// sub-step.
//
// The grid, constellations, resampling and springl splitting are AdaptiveDoubleweaveLevelSet2D's:
// the band around the springls is kept at maxDepth, and each constellation is filled and
// contracted against its own rebuilt contour.
class SemiLagrangianDoubleweaveLevelSet2D : public AdaptiveDoubleweaveLevelSet2D
{
public:
    struct SemiLagrangianTuning {
        float capsuleCells = 1.5f;  // capsule radius of the reconstruction, in cells
        // Floor on the capsule radius, in base pixels (see SignFieldParams::minBandPixels).
        float minBandPixels = 0.3f;
        // Largest relative stretch of any springl (or pair of neighboring springls) per sub-step.
        float maxStrain = 0.5f;
        // Optional cap on displacement per sub-step, in units of the springl's own scale.
        float maxStepCells = std::numeric_limits<float>::infinity();
        // Closed contour loops enclosing less than minLoopCells * h^2 (h the finest cell along the
        // loop) are erased: sub-resolution blobs and pockets vanish, as surfaces tear in Bargteil
        // et al. 2006, instead of keeping their springls alive.
        float minLoopCells = 4.0f;
    };
    SemiLagrangianTuning slTuning;

    SemiLagrangianDoubleweaveLevelSet2D();
    bool step() override;

protected:
    // Reconstructs mGrid from the included springls (no PDE), then re-extracts the contour.
    void evolveGrid() override;
    // The level set is rebuilt from the springls as advected, before relaxation: relaxation pulls
    // arm vertexes toward their neighbors and can open holes in the band that advection did not.
    bool rebuildsLevelSetBeforeRelax() const override { return true; }
    // Strain-limited step: maxStrain / max strain, capped by maxStepCells; 1 time unit when nothing
    // else bounds it (advect() still applies the schedule's cap). Strain is taken within each
    // springl (|v(v1) - v(v0)| / |v1 - v0|), between attraction neighbors, and between each
    // particle and the nearest springl of the other constellation, as |k1_i - k1_j| / gap. Pairs
    // count only when same-facing; the gap is floored at kMinPairGapScale * scale.
    float advectionTimeStep(const Points2 &k1, float maxStep) const override;
    // Erases closed curves of the current contour that enclose less than minLoopCells * h^2: grid
    // nodes inside such a loop on its interior side take the sign outside it (same |phi|). Returns
    // how many loops were erased; the caller re-extracts the contour.
    int eraseTinyLoops();

    // Springls seeded or filled from contour edges shorter than kMinSeedLengthCells * scale are
    // dropped: marching squares cuts tiny edges near grid nodes, and their direction (so their
    // normal) is noise. The gap a tiny edge leaves is covered by its neighbors.
    static constexpr float kMinSeedLengthCells = 0.25f;
    // Strain pairs measure the particle gap at least this many springl scales.
    static constexpr float kMinPairGapScale = 0.25f;
    size_t dropDegenerateSpringls(size_t from);
    void seedSpringls() override;
    int fill() override;
};
}  // namespace springls
