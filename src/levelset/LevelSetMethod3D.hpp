#pragma once
#include <springl/AdaptiveSpringLevelSet3D.hpp>
#include <springl/SpringLevelSet3D.hpp>

namespace springls
{
// The basic level set method in 3D [Osher and Sethian, J. Comput. Phys. 1988]: the narrow-band
// level set (LevelSet3D, the same one the spring level set evolves toward its springls) is moved by
// the velocity field itself, phi_t + v . grad(phi) = w kappa, with first-order upwind advection
// and forward Euler in time; 3D counterpart of LevelSetMethod2D. The iso-surface is extracted
// every step.
//
// Derives from SpringLevelSet3D for its velocity, schedule, bookkeeping and checkpoints (so the
// Enright harness runs it like the other methods) but never creates springls.
class LevelSetMethod3D : public SpringLevelSet3D
{
public:
    explicit LevelSetMethod3D(ElementType elementType = ElementType::Quad);
    bool init() override;
    bool step() override;

protected:
    void extractSurface();
};

// The basic level set method on the block-adaptive octree (AdaptiveGrid3D, the grid adaptive
// springls use): each sub-step the band is re-fitted around the current iso-surface, then the
// corners near the interface move with the flow by upwind advection (AdaptiveGrid3D::evolve).
//
// Derives from AdaptiveSpringLevelSet3D for the grid set-up, band adaptation, surface extraction
// and checkpoints (so the Enright harness and renderer treat it as an adaptive method).
class AdaptiveLevelSetMethod3D : public AdaptiveSpringLevelSet3D
{
public:
    explicit AdaptiveLevelSetMethod3D(ElementType elementType = ElementType::Quad);
    bool init() override;
    bool step() override;
};
}  // namespace springls
