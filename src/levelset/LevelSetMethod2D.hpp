#pragma once
#include <springl/AdaptiveSpringLevelSet2D.hpp>
#include <springl/SpringLevelSet2D.hpp>

namespace springls
{
// The basic level set method [Osher and Sethian, J. Comput. Phys. 1988]: a single interface whose
// narrow-band level set (LevelSet2D, the same one the spring level set evolves toward its springls)
// is moved by the velocity field itself, phi_t + v . grad(phi) = w kappa, with first-order upwind
// advection and forward Euler in time. The iso-contour is extracted every step. It is the Eulerian
// baseline the springl methods are compared against.
//
// Derives from SpringLevelSet2D for its velocity, schedule and bookkeeping (so the Enright harness
// runs it like the other methods) but never creates springls.
class LevelSetMethod2D : public SpringLevelSet2D
{
public:
    LevelSetMethod2D();
    bool init() override;
    bool step() override;
};

// The basic level set method on the block-adaptive quadtree (AdaptiveGrid2D, the grid adaptive
// springls use): each sub-step the band is re-fitted around the current contour (finest leaves
// within the band margin of it, coarser leaves elsewhere), then the corners near the interface
// move with the flow by upwind advection (AdaptiveGrid2D::evolve).
//
// Derives from AdaptiveSpringLevelSet2D for the grid set-up, band adaptation and contour
// extraction (so the Enright harness and renderer treat it as an adaptive method).
class AdaptiveLevelSetMethod2D : public AdaptiveSpringLevelSet2D
{
public:
    AdaptiveLevelSetMethod2D();
    bool init() override;
    bool step() override;

protected:
    // Moves the interface with the flow for time dt from time t.
    void evolve(double t, float dt);
};
}  // namespace springls
