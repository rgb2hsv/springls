#pragma once
#include <springl/MultiSpringLevelSet2D.hpp>

#include <array>

namespace springls
{
// Multi-object DoubleWeave springls on a uniform grid: DweaveLevelSet2D's two staggered
// constellations on MultiSpringLevelSet2D's MOGAC label + unsigned distance band.
//
// Every springl carries a region label and a constellation. The primal constellation seeds one
// springl per edge of each labeled curve, the dual one springl per curve vertex (MakeChordSpringl);
// both take the curve's label. Attraction, fill, contraction and tracking act within one label and
// one constellation. All springls drive the combined band (the output); each constellation also
// keeps its own band, evolved from its own springls alone, against which it is filled and
// contracted, so a hole in one constellation shows even where the other still holds the region.
class MultiDweaveLevelSet2D : public MultiSpringLevelSet2D
{
public:
    MultiDweaveLevelSet2D();
    bool init() override;
    size_t countSpringls(Constellation constellation) const;

protected:
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    // One constellation's band and labeled contour, evolved from that constellation's springls only.
    struct ConstellationLevelSet {
        MultiNarrowBand2D band;
        ImageF unsignedLevelSet;
        Points2 vertexLocations;
        Curves vertexIndexes;
        std::vector<int> curveLabels;
        bool valid = false;
    };
    std::array<ConstellationLevelSet, kConstellationCount> mConstellations;
    // Constellation swapped into the base-class members, or -1 for the combined one.
    int mActiveConstellation = -1;
    std::vector<uint8_t> mOldConstellations;

    bool inConstellation(size_t springl, size_t w) const { return mManifold.springlConstellations[springl] == w; }
    // Exchanges the base-class band / contour with constellation w's.
    void swapConstellation(size_t w);
    void ensureConstellationLevelSets();
    // Unsigned distance to constellation w's springls (constellation w must be swapped in).
    void updateConstellationUnsignedLevelSet(size_t w);
    void evolveConstellationLevelSets();
    size_t appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, int label, Constellation constellation);

    void seedSpringls() override;
    void clearSpringls() override;
    void compactSpringls(const std::vector<uint8_t> &keep) override;
    void snapshotForTracking() override;
    bool canTrackFrom(size_t newIndex, size_t oldIndex) const override;
    bool canAttract(size_t springl, size_t other) const override;
    int fill() override;
    int contract() override;
    void afterLevelSetUpdate() override { evolveConstellationLevelSets(); }
    bool refinesContourWith(size_t particle) const override;
};
}  // namespace springls
