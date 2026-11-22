#pragma once
#include <springl/AdaptiveMultiSpringLevelSet2D.hpp>

#include <array>

namespace springls
{
// Multi-object DoubleWeave springls on the labeled block-adaptive quadtree:
// AdaptiveDweaveLevelSet2D's two constellations on AdaptiveMultiSpringLevelSet2D's labeled
// grid. The combined grid (all springls) is the output and decides springl splitting; each
// constellation also keeps its own labeled grid, banded around and evolved from its own springls,
// against which it is filled and contracted. Springls interact only within one region label and
// one constellation.
class AdaptiveMultiDweaveLevelSet2D : public AdaptiveMultiSpringLevelSet2D
{
public:
    AdaptiveMultiDweaveLevelSet2D();
    bool init() override;
    size_t countSpringls(Constellation constellation) const;
    const AdaptiveGrid2D &getConstellationGrid(Constellation c) const { return mStates[static_cast<size_t>(c)].grid; }

protected:
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    struct ConstellationState {
        AdaptiveGrid2D grid;
        Points2 vertexLocations;
        Curves vertexIndexes;
        std::vector<int> curveLabels;
        std::vector<float> contourScales;
    };
    std::array<ConstellationState, kConstellationCount> mStates;
    int mActiveConstellation = -1;
    std::vector<uint8_t> mOldConstellations;

    bool inConstellation(size_t springl, size_t c) const { return mManifold.springlConstellations[springl] == c; }
    void swapState(size_t c);
    void evolveConstellations();
    size_t appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale, int label, Constellation c);

    void seedSpringls() override;
    void clearSpringls() override;
    void compactSpringls(const std::vector<uint8_t> &keep) override;
    void snapshotForTracking() override;
    bool canTrackFrom(size_t newIndex, size_t oldIndex) const override;
    bool canAttract(size_t springl, size_t other) const override;
    int fill() override;
    int contract() override;
    void evolveLevelSetAndContour() override;
    bool refinesContourWith(size_t particle) const override { return levelSetIncludes(particle); }
    bool levelSetIncludes(size_t springl) const override;
    void splitSpringl(size_t index) override;
};
}  // namespace springls
