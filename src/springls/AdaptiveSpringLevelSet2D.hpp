#pragma once
#include <springls/AdaptiveGrid2D.hpp>
#include <springls/SegmentTree2D.hpp>
#include <springls/SpringLevelSet2D.hpp>

#include <functional>

namespace springls
{
// Spring level set on a block-adaptive quadtree (AdaptiveGrid2D).
//
// The quadtree stores a narrow band at a single, finest resolution around the springls (as an
// OpenVDB grid does, and as AdaptiveSpringLevelSet3D does): before each level-set update, every
// leaf within the band margin (see bandCells) of a springl particle is refined to
// adaptiveTuning.maxDepth, and parents with no particle within the margin coarsen (one depth per
// update). The margin covers how far springls move in a sub-step plus the active band, so the zero
// level set always lies in finest leaves; coarser leaves only fill the space away from the
// interface. At init, the band follows the initial interface instead. Springls are created at the
// finest cell size; any springl that a refinement leaves larger than its leaf is split in two.
class AdaptiveSpringLevelSet2D : public SpringLevelSet2D
{
public:
    struct AdaptiveTuning {
        int maxDepth = 3;        // band cells are 2^-maxDepth base pixels
        float bandWidth = 3.0f;  // phi is clamped to +-bandWidth (base pixels)
        // Leaves within bandCells finest cells plus one sub-step of motion (tuning.maxStepPixels) of a
        // springl particle are at maxDepth: enough for the zero layer, its three rebuilt distance
        // layers and a cell of slack to stay in finest leaves until the band is re-fitted, at the
        // next sub-step.
        float bandCells = 5.0f;
        // After each evolution, corners within a cell of a springl that lie on the wrong side of it
        // take its signed distance (particle-level-set style one-sided correction).
        bool springlCorrection = true;
    };
    AdaptiveTuning adaptiveTuning;

    AdaptiveSpringLevelSet2D();
    bool step() override;

    const AdaptiveGrid2D &getGrid() const { return mGrid; }
    // Largest |phi| at a springl particle, in local cell sizes (a diagnostic).
    float maxTrackingError() const;

protected:
    AdaptiveGrid2D mGrid;
    std::vector<float> mContourScales;
    // Whether refining mGrid splits springls that no longer fit their leaf.
    bool mSplitOnRefine = true;

    void initLevelSet() override;
    float contourScale(uint32_t vertex) const override { return (vertex < mContourScales.size()) ? mContourScales[vertex] : 1.0f; }
    // Attraction reaches as far as the level set is resolved where the vertex currently is.
    float attractionScale(const Vec2 &position, size_t) const override { return mGrid.cellSize(position); }
    void evolveLevelSetAndContour() override;
    void solveContourForFill() override;
    void afterResample() override {}

    // Refines leaves for which near(block) holds to maxDepth, then coarsens (one depth) parents
    // none of whose children are near.
    void adaptBand(const std::function<bool(const AdaptiveGrid2D::Block &)> &near);
    // The band around the particles of the springls that drive mGrid.
    void adaptToParticles();
    // Distance (base pixels) within which leaves are kept at maxDepth.
    float bandMargin() const { return tuning.maxStepPixels + adaptiveTuning.bandCells * std::ldexp(1.0f, -adaptiveTuning.maxDepth); }
    // Segments of the springls that drive mGrid (levelSetIncludes()), ids = springl indexes.
    SegmentTree2D makeSpringlTree() const;
    // Pull of every active node toward its nearest springl (cell units), for activeLeaves().
    void updateGridVelocities(const SegmentTree2D &springlTree, std::vector<AdaptiveGrid2D::NodeVelocities> &velocities);
    // The one-sided springl correction (see AdaptiveTuning::springlCorrection).
    int correctFromSpringls(const SegmentTree2D &springlTree);
    // Sparse-field evolution toward the springls, then contour extraction.
    virtual void evolveGrid();
    void extractGridContour();
    void splitOversizedSpringls();
    virtual void splitSpringl(size_t index);
    // Merges pairs of neighboring springls (the end vertex of one attracted to the start vertex of
    // the other) that the flow compressed until together they span no more than one springl of
    // their size class, so springl density falls again where the contour shrinks. The merged
    // springl keeps both outer ends, so it opens no gap for fill to refill. Returns the merge count.
    size_t mergeCompressedSpringls();
    // Springls whose particles and segments drive mGrid (default: all).
    virtual bool levelSetIncludes(size_t) const { return true; }
};

// Doubleweave springls on a block-adaptive quadtree: the combined grid (all springls) is the
// output surface and decides springl splitting; each constellation also keeps its own grid,
// banded around and evolved from its own springls, against which it is filled and contracted.
class AdaptiveDoubleweaveLevelSet2D : public AdaptiveSpringLevelSet2D
{
public:
    AdaptiveDoubleweaveLevelSet2D();
    bool init() override;
    size_t countSpringls(Constellation constellation) const;
    const AdaptiveGrid2D &getConstellationGrid(Constellation c) const { return mStates[static_cast<size_t>(c)].grid; }

protected:
    struct ConstellationState {
        AdaptiveGrid2D grid;
        Points2 vertexLocations;
        Curves vertexIndexes;
        std::vector<float> contourScales;
    };
    std::array<ConstellationState, kConstellationCount> mStates;
    int mActiveConstellation = -1;
    std::vector<uint8_t> mOldConstellations;

    void swapState(size_t c);
    void evolveConstellations();
    size_t appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale, Constellation c);

    void seedSpringls() override;
    void clearSpringls() override;
    void compactSpringls(const std::vector<uint8_t> &keep) override;
    void snapshotForTracking() override;
    bool canTrackFrom(size_t newIndex, size_t oldIndex) const override;
    bool canAttract(size_t springl, size_t other) const override { return mManifold.springlConstellations[springl] == mManifold.springlConstellations[other]; }
    int fill() override;
    int contract() override;
    void evolveLevelSetAndContour() override;
    bool refinesContourWith(size_t particle) const override { return levelSetIncludes(particle); }
    bool levelSetIncludes(size_t springl) const override;
    void splitSpringl(size_t index) override;
};
}  // namespace springls
