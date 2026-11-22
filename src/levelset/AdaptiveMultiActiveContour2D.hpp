#pragma once
#include <springl/AdaptiveMultiSpringLevelSet2D.hpp>

namespace springls
{
// MOGAC without springls on the labeled block-adaptive quadtree: the adaptive counterpart of
// MultiActiveContour2D. Corners carry a region label and the unsigned distance to the nearest
// region boundary; each sub-step the band is re-fitted around the current labeled contour (finest
// leaves within the band margin of it, coarser leaves elsewhere), and the corners near the
// boundaries move by upwind advection with the flow itself plus curvature (AdaptiveGrid2D::
// evolveLabeled). Never creates springls.
//
// Derives from AdaptiveMultiSpringLevelSet2D for the labeled grid set-up, band adaptation and
// contour extraction (and so the Enright harness and renderer treat it as an adaptive method).
class AdaptiveMultiActiveContour2D : public AdaptiveMultiSpringLevelSet2D
{
public:
    AdaptiveMultiActiveContour2D();
    bool init() override;
    bool step() override;

protected:
    // One sub-step: moves the region boundaries with the flow for time dt from time t.
    void evolve(double t, float dt);
    // Largest flow speed (pixels per unit time) at the active corners at time t.
    float maxBandSpeed(double t);
};
}  // namespace springls
