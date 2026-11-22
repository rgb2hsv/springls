#pragma once
#include <levelset/MultiNarrowBand2D.hpp>
#include <springl/SpringLevelSet2D.hpp>

namespace springls
{
// Multi-object geodesic active contours (MOGAC) without springls [Lucas, Kazhdan and Taylor,
// MICCAI 2012]: the label + unsigned distance narrow band is moved by the velocity field itself
// (upwind advection) and curvature, and labeled curves are extracted every step. It is the
// Eulerian baseline the multi-object springls methods are compared against.
//
// Derives from SpringLevelSet2D for its velocity, schedule and bookkeeping (so the Enright
// harness runs it like the other methods) but never creates springls.
class MultiActiveContour2D : public SpringLevelSet2D
{
public:
    MultiActiveContour2D();
    bool init() override;
    bool step() override;
    const MultiNarrowBand2D &getBand() const { return mMultiBand; }

protected:
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    MultiNarrowBand2D mMultiBand;
    std::vector<int> mObjectLabels;

    // One band update moving the interface for time dt from time t.
    void evolve(double t, float dt);
    void extractContour();
};
}  // namespace springls
