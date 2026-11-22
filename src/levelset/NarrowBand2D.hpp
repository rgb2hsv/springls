#pragma once
#include <core/Image.hpp>

#include <functional>
#include <vector>

namespace springls
{
// Narrow-band signed level set on a uniform grid (phi < 0 inside, in pixels), updated with the
// sparse-field method [Whitaker, "A Level-Set Approach to 3D Reconstruction from Range Data",
// IJCV 1998].
//
// Pixels with |phi| <= kMaxDistance are active. update() applies a per-active-pixel speed
// (computed by the caller into delta()) to the zero layer, rebuilds kLayers distance layers
// around it, and grows / shrinks the band.
class NarrowBand2D
{
public:
    static constexpr float kMaxDistance = 3.5f;
    static constexpr int kLayers = 3;

    // Initializes from a signed distance field (clamped to the band).
    void initialize(const ImageF &signedDistance);
    bool empty() const { return mPhi.empty(); }

    const ImageF &phi() const { return mPhi; }
    // Values of the previous update, used for the speed stencil.
    const ImageF &previous() const { return mSwap; }
    const std::vector<Vec2i> &active() const { return mActive; }
    // Speed of each active pixel, filled by the caller before update().
    std::vector<float> &delta() { return mDelta; }

    // phi += timeStep * clamp(delta, -1, 1) on the zero layer, then rebuilds the distance layers
    // and the band.
    void update(float timeStep);
    // Measured correction: measure(i, j, d) may supply a signed distance d for active pixel (i, j);
    // where its sign disagrees with phi, phi (and the previous values) take d. Returns how many
    // pixels changed.
    int correctSigns(const std::function<bool(int i, int j, float &d)> &measure);

    // cereal serialization (checkpoints).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(mPhi, mSwap, mInBand, mActive, mDelta);
    }

private:
    ImageF mPhi;
    ImageF mSwap;
    std::vector<uint8_t> mInBand;  // per pixel, whether it is in mActive
    std::vector<Vec2i> mActive;
    std::vector<float> mDelta;

    void rebuild();
    void updateDistanceLayer(int i, int j, int band);
    void plug(int i, int j);
    void deleteElements();
    void addElements();
};
}  // namespace springls
