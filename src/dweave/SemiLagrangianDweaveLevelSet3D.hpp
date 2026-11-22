#pragma once
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <limits>

namespace springls
{
// Semi-Lagrangian DoubleWeave springls on the adaptive octree; the 3D counterpart of
// SemiLagrangianDweaveLevelSet2D.
//
// Springls are advected freely, and every grid (the combined one and each constellation's) is
// rebuilt from its springls after each sub-step, before relaxation, with SpringlSignField3D: the
// min unsigned distance to the springls, signed by topology (an exterior flood around the
// springls' capsules and an upwind sweep into them), with no springl normals involved. The level
// set never moves on its own, so there is no level-set PDE and no CFL limit; the sub-step is
// limited by how much the flow stretches the springls (strain) and by the advection schedule.
class SemiLagrangianDweaveLevelSet3D : public AdaptiveDweaveLevelSet3D
{
public:
    struct SemiLagrangianTuning {
        float capsuleCells = 1.5f;  // capsule radius of the reconstruction, in finest cells
        // Largest relative stretch of any springl (or pair of neighboring springls) per sub-step.
        float maxStrain = 0.5f;
        // Optional cap on displacement per sub-step, in units of the springl's own scale.
        float maxStepCells = std::numeric_limits<float>::infinity();
        // Closed surface components enclosing less than minComponentCells * h^3 (h the finest
        // cell on the component) are erased: sub-resolution blobs and bubbles vanish instead of
        // keeping their springls alive.
        float minComponentCells = 8.0f;
    };
    SemiLagrangianTuning slTuning;

    explicit SemiLagrangianDweaveLevelSet3D(ElementType elementType = ElementType::Quad);
    bool step() override;

protected:
    // Rebuilds mGrid from the included springls (no PDE), then re-extracts the iso-surface.
    void evolveGrid() override;
    bool rebuildsLevelSetBeforeRelax() const override { return true; }
    // The band is fitted after advection, right before each rebuild, so it needs no slack for a
    // sub-step of motion (the CFL-limited level set does): bandCells finest cells, which hold the
    // capsules (capsuleCells, or the 0.3-voxel floor) plus a springl's extent.
    float bandMargin() const override { return adaptiveTuning.bandCells * std::ldexp(1.0f, -adaptiveTuning.maxDepth); }
    // Strain-limited step: maxStrain / max strain, capped by maxStepCells; 1 time unit when nothing
    // else bounds it (advect() still applies the schedule's cap). Strain is taken within each
    // springl (|v(vertex) - v(particle)| / arm length), between attraction neighbors, and between
    // each particle and the nearest springl of the other constellation, as |k1_i - k1_j| / gap,
    // for same-facing pairs only, with the gap floored at kMinPairGapScale * scale.
    float advectionTimeStep(const Points3 &k1, float maxStep) const override;
    // Erases closed components of the iso-surface enclosing less than minComponentCells * h^3:
    // grid nodes inside such a component on its interior side take the outside sign (same |phi|).
    // Returns how many were erased; the caller re-extracts the surface.
    int eraseTinyComponents();
    static constexpr float kMinPairGapScale = 0.25f;
    // |phi| on the finest leaves is the true distance out to this many finest cells, then clamped.
    static constexpr float kDistanceCells = 4.0f;
};
}  // namespace springls
