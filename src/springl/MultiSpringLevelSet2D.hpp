#pragma once
#include <levelset/MultiNarrowBand2D.hpp>
#include <springl/SpringLevelSet2D.hpp>

namespace springls
{
// Multi-object spring level sets on a uniform grid: springls carry a region label and the topology
// comes from a MOGAC label + unsigned distance band (MultiNarrowBand2D) instead of one signed
// level set. Every sub-step the band is pulled toward the unsigned distance field of all springls,
// each candidate label with its own upwind stencil (as wedge's MultiSpringLevelSet2D); labeled
// curves are extracted from it, and springls resample against their own region's curves only
// (the label rules of SpringLevelSet2D). A boundary between two regions carries two springl
// layers, one per side.
class MultiSpringLevelSet2D : public SpringLevelSet2D
{
public:
    MultiSpringLevelSet2D();
    const MultiNarrowBand2D &getBand() const { return mMultiBand; }

protected:
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    MultiNarrowBand2D mMultiBand;
    std::vector<int> mObjectLabels;

    Vec2i levelSetSize() const override { return Vec2i(mMultiBand.labels().width(), mMultiBand.labels().height()); }
    void initLevelSet() override;
    void evolveLevelSetAndContour() override;
    void solveContourForFill() override;

    void extractContour();
    // Particle-sample correction of the band from the springls for which include(n) holds (all if
    // empty): band pixels beside a springl, on its inner side, take its region.
    void correctFromSpringls(const std::function<bool(size_t)> &include = nullptr);
    // One band update: every candidate label moves toward the springls (background: curvature only).
    void updateMultiLevelSet(float maxStep = 0.5f);
};
}  // namespace springls
