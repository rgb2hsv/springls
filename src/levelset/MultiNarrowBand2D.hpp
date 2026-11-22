#pragma once
#include <core/Image.hpp>

#include <array>
#include <functional>
#include <limits>
#include <vector>

namespace springls
{
// Narrow band of a multi-object level set on a uniform grid, in the MOGAC representation
// [Lucas, Kazhdan and Taylor, "Multi-Object Geodesic Active Contours (MOGAC)", MICCAI 2012]: every
// pixel stores a region label and the unsigned distance (pixels) to the nearest region boundary.
// The signed level set of region l is -d inside l and +d outside, so regions can never overlap or
// leave a vacuum.
//
// update() moves the zero layer (d <= 1/2) by per-candidate speeds filled by the caller: each
// pixel may take its own label or one of its four neighbors'; it takes the label whose updated
// signed value is smallest, with half the gap to the second smallest as its new distance. Distance
// layers are then rebuilt, isolated pixels are plugged and the band grows / shrinks, as the
// sparse-field method does for one level set (see NarrowBand2D).
class MultiNarrowBand2D
{
public:
    static constexpr float kMaxDistance = 3.5f;
    static constexpr int kLayers = 3;
    static constexpr int kCandidates = 5;  // own label + the 4-neighbors'
    static constexpr int kNoLabel = std::numeric_limits<int>::min();
    // Candidate labels of an active pixel: distinct, unused slots kNoLabel.
    using Candidates = std::array<int, kCandidates>;
    using Speeds = std::array<float, kCandidates>;

    // distance is unsigned (pixels); it is clamped to kMaxDistance + 0.5.
    void initialize(const ImageI &labels, const ImageF &distance);
    bool empty() const { return mLabels.empty(); }

    const ImageI &labels() const { return mLabels; }
    const ImageF &distance() const { return mDistance; }
    // Values of the previous update, used for the speed stencil.
    const ImageI &previousLabels() const { return mPrevLabels; }
    const ImageF &previousDistance() const { return mPrevDistance; }
    float signedValue(int i, int j, int label) const { return (mLabels.at(i, j) == label) ? -mDistance.at(i, j) : mDistance.at(i, j); }
    float previousSignedValue(int i, int j, int label) const { return (mPrevLabels.at(i, j) == label) ? -mPrevDistance.at(i, j) : mPrevDistance.at(i, j); }
    // Bilinear interpolation of region `label`'s signed values (border clamped).
    float signedValue(const Vec2 &p, int label) const;

    // Speed stencils on the previous signed values of region `label` at pixel (i, j):
    // g . grad(phi_l) with upwind differences [Osher and Sethian 1988], and the curvature term
    // weight * kappa_l clamped to +-10 (as LevelSet2D::speed computes them).
    float previousUpwindAdvection(int i, int j, int label, const Vec2 &g) const;
    float previousCurvatureTerm(int i, int j, int label, float weight) const;

    const std::vector<Vec2i> &active() const { return mActive; }
    const std::vector<Candidates> &candidates() const { return mCandidates; }
    // Speed (cells per unit time step) of each candidate of each active pixel, filled by the caller.
    std::vector<Speeds> &delta() { return mDelta; }

    // Moves the zero layer by timeStep * clamp(delta, -1, 1), then rebuilds the distance layers
    // and the band.
    void update(float timeStep);

    // Measured correction: measure(i, j, label, d) may report that active pixel (i, j) lies inside
    // region `label` at distance d from its boundary; a pixel with another label takes it (current
    // and previous values alike). Candidates are refreshed. Returns how many pixels changed.
    int correctLabels(const std::function<bool(int i, int j, int &label, float &d)> &measure);

    // cereal serialization (checkpoints).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(mLabels, mPrevLabels, mDistance, mPrevDistance, mInBand, mActive, mCandidates, mDelta);
    }

private:
    ImageI mLabels, mPrevLabels;
    ImageF mDistance, mPrevDistance;
    std::vector<uint8_t> mInBand;
    std::vector<Vec2i> mActive;
    std::vector<Candidates> mCandidates;
    std::vector<Speeds> mDelta;

    void rebuild();
    void refreshCandidates();
    void deleteElements();
    void addElements();
};

// Result of the MOGAC two-smallest rule at one node.
struct MogacResult {
    int label = MultiNarrowBand2D::kNoLabel;
    float distance = 0.0f;
};
// Given the node's current label and each candidate's updated signed value, the label the node
// takes and its new unsigned distance. Ties go to the current label, then to the smaller label.
MogacResult MogacUpdate(int currentLabel, const MultiNarrowBand2D::Candidates &labels, const MultiNarrowBand2D::Speeds &updatedValues);
}  // namespace springls
