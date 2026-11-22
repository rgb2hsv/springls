#pragma once
#include <springl/AdaptiveSpringLevelSet2D.hpp>

namespace springls
{
// Multi-object springls on the block-adaptive quadtree: the grid is labeled (AdaptiveGrid2D in
// MOGAC mode: a region label and unsigned distance per corner), evolved toward the springls with
// the two-smallest rule, and its labeled contour drives resampling against each region's own
// springls. Band adaptation, springl splitting / merging and contour scales are those of
// AdaptiveSpringLevelSet2D.
class AdaptiveMultiSpringLevelSet2D : public AdaptiveSpringLevelSet2D
{
public:
    AdaptiveMultiSpringLevelSet2D();

protected:
    void initLevelSet() override;
    void evolveGrid() override;
    void extractGridContour() override;
};
}  // namespace springls
