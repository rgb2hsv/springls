#pragma once
#include <core/Volume.hpp>

#include <functional>
#include <string>
#include <vector>

namespace springls
{
class VdbWriter;
class VdbReader;

// Narrow-band signed level set on a uniform 3D grid (phi < 0 inside, in voxels), updated with the
// sparse-field method [Whitaker, IJCV 1998]; 3D counterpart of NarrowBand2D.
//
// Voxels with |phi| <= kMaxDistance are active. update() applies a per-active-voxel speed
// (computed by the caller into delta()) to the zero layer, rebuilds kLayers distance layers
// around it, and grows / shrinks the band.
class NarrowBand3D
{
public:
    static constexpr float kMaxDistance = 3.5f;
    static constexpr int kLayers = 3;

    // Initializes from a signed distance field (clamped to the band).
    void initialize(const VolumeF &signedDistance);
    bool empty() const { return mPhi.empty(); }

    const VolumeF &phi() const { return mPhi; }
    // Values of the previous update, used for the speed stencil.
    const VolumeF &previous() const { return mSwap; }
    const std::vector<Vec3i> &active() const { return mActive; }
    // Speed of each active voxel, filled by the caller before update().
    std::vector<float> &delta() { return mDelta; }

    // phi += timeStep * clamp(delta, -1, 1) on the zero layer, then rebuilds the distance layers
    // and the band.
    void update(float timeStep);
    // Measured correction: measure(i, j, k, d) may supply a signed distance d for active voxel
    // (i, j, k); where its sign disagrees with phi, phi (and the previous values) take d. Returns
    // how many voxels changed.
    int correctSigns(const std::function<bool(int i, int j, int k, float &d)> &measure);

    // cereal serialization (cached state).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(mPhi, mSwap, mInBand, mActive, mDelta);
    }
    // Checkpoints split in two: the level sets (phi and the previous update) as VDB volumes "<prefix>phi" and
    // "<prefix>previous" (see WriteVolumeVdb), and the band bookkeeping with cereal (serializeBand).
    bool writeVdb(VdbWriter &writer, const std::string &prefix, std::string &error) const;
    bool readVdb(const VdbReader &reader, const std::string &prefix, std::string &error);
    template <class Archive>
    void serializeBand(Archive &ar)
    {
        ar(mInBand, mActive, mDelta);
    }

private:
    VolumeF mPhi;
    VolumeF mSwap;
    std::vector<uint8_t> mInBand;  // per voxel, whether it is in mActive
    std::vector<Vec3i> mActive;
    std::vector<float> mDelta;

    void rebuild();
    void updateDistanceLayer(int i, int j, int k, int band);
    void plug(int i, int j, int k);
    void deleteElements();
    void addElements();
};
}  // namespace springls
