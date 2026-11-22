#include <springl/AdaptiveMultiSpringLevelSet2D.hpp>

#include <spatial/SegmentTree2D.hpp>
#include <springl/SpringlRegionProbe2D.hpp>

namespace springls
{
using Block = AdaptiveGrid2D::Block;

AdaptiveMultiSpringLevelSet2D::AdaptiveMultiSpringLevelSet2D()
{
    mName = "Adaptive Multi-Object Springls";
    mLabeled = true;
}

void AdaptiveMultiSpringLevelSet2D::initLevelSet()
{
    mScaledSpringls = true;
    mBand = LevelSet2D();  // the uniform band is unused
    mGrid.initializeLabeled(mInitialLabels, mInitialLabelDistance, adaptiveTuning.maxDepth, adaptiveTuning.bandWidth);
    // No springls yet: the band follows the initial region boundaries (phi is their distance).
    const float margin = bandMargin();
    adaptBand([margin](const Block &block) {
        float minDistance = block.tileValue;
        if (!block.isTile()) {
            minDistance = std::numeric_limits<float>::max();
            for (float v : block.phi) minDistance = std::min(minDistance, v);
        }
        return minDistance < margin + std::sqrt(2.0f) * block.h;
    });
    extractGridContour();
    // Springls are seeded one per contour segment: even out the spacing of the grid crossings (as
    // MultiSpringLevelSet2D does), keeping region junction corners in place.
    evenInitialContourSpacing();
}

void AdaptiveMultiSpringLevelSet2D::extractGridContour()
{
    mGrid.extractLabeledContour(mManifold.vertexLocations, mManifold.vertexIndexes, mManifold.curveLabels, mContourScales, &mContourJunctions);
}

void AdaptiveMultiSpringLevelSet2D::evolveGrid()
{
    std::vector<AdaptiveGrid2D::NodeVelocities> velocities;  // transient: active blocks only
    const SegmentTree2D springlTree = makeSpringlTree();
    updateGridVelocities(springlTree, velocities);
    mGrid.evolveLabeled(tuning.signedEvolveIterations, mCurvatureWeight, backgroundCurvatureWeight, velocities);
    if (adaptiveTuning.springlCorrection) {
        // A corner within a cell of a springl, beside it and on its inner side, lies in its region.
        // Each region's springls are searched separately (see SpringlRegionProbe2D).
        mManifold.updateNormals();
        const SpringlRegionProbe2D probe(mManifold, [this](size_t n) { return levelSetIncludes(n); });
        mGrid.correctLabels([&](const Vec2 &q, float h, int &label, float &d) { return probe.inside(q, h, label, d); });
    }
    extractGridContour();
}
}  // namespace springls
