#pragma once
#include <springls/Image.hpp>
#include <springls/Manifold2D.hpp>
#include <springls/NarrowBand2D.hpp>
#include <springls/SpringlHashGrid2D.hpp>

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace springls
{
// Spring level sets in 2D [Lucas, Kazhdan and Taylor, "Spring Level Sets: A Deformable Model
// Representation to Provide Interoperability between Meshes and Level Sets", IEEE TVCG 2012].
//
// The interface is carried by springls: small line segments (two arm vertexes around a
// particle) advected by the flow. A narrow-band level set is evolved toward the unsigned distance
// field of the springls every sub-step; its iso-contour defines the topology and is used to
// resample the springls (contract springls far from it, fill gaps along it).
//
// Positions are in level-set pixels. Typical use:
//   sim.setInitialLevelSet(phi); sim.setVelocityField(v, 1); sim.init(); while (sim.step()) {}
class SpringLevelSet2D
{
public:
    static constexpr float kParticleRadius = 0.05f;
    static constexpr float kSpringConstant = 0.3f;
    static constexpr float kExtent = 0.5f;  // springl half-size, in cells
    static constexpr float kMaxDistance = 8.0f;
    static constexpr float kSharpness = 5.0f;
    static constexpr float kMinEdgeLength = 0.001f;

    // Runtime-tunable parameters. Defaults were tuned on the 2D Enright test.
    struct Tuning {
        // Relaxation.
        float armRestLength = 2.0f * kParticleRadius;           // rest length of each arm spring
        float neighborRestSeparation = 2.0f * kParticleRadius;  // target gap between neighboring arm vertexes
        float springConstant = kSpringConstant;
        float sharpness = kSharpness;  // strength of neighbor attraction
        float nearestNeighborDistance = 0.5f;
        int relaxIterations = 4;
        float relaxTimeStep = 0.01f;
        // Resampling (fill / contract).
        float fillDistance = 0.5f;
        float contractDistance = 1.2f;
        float maxArmLength = 1.0f;
        // Level set and time stepping.
        int signedEvolveIterations = static_cast<int>(std::ceil(kMaxDistance * 1.9f));
        float unsignedMaxDistance = 6.0f * kExtent;
        float maxStepPixels = 0.333333f;   // max particle displacement per advection sub-step
        int advectionSubstepsPerStep = 3;  // advection sub-steps (each with a level-set update) per step()
        int refineIterations = 4;
        float refineProximity = 2.0f;
        float refineStepSize = 0.9f;
    };
    enum class TemporalScheme {
        FirstOrder,
        RK4
    };
    // Time-varying advection: before each sub-step the vector-field weight is set to weight(t);
    // sub-steps are capped by maxTimeStep and clamped so time never passes endTime.
    struct AdvectionSchedule {
        std::function<float(double)> weight;
        double endTime = std::numeric_limits<double>::infinity();
        double maxTimeStep = std::numeric_limits<double>::infinity();
    };

    Tuning tuning;
    TemporalScheme temporalScheme = TemporalScheme::RK4;
    // Prints progress to std::cerr every kLogInterval steps.
    bool verbose = false;
    static constexpr uint64_t kLogInterval = 10;

    SpringLevelSet2D();
    virtual ~SpringLevelSet2D() = default;
    const std::string &name() const { return mName; }

    // Signed distance (pixels, negative inside) of the initial interface.
    void setInitialLevelSet(const ImageF &signedDistance) { mInitialLevelSet = signedDistance; }
    // Velocity field in pixels per unit time, scaled by `weight` (or the schedule weight).
    void setVelocityField(const ImageV2 &field, float weight)
    {
        mVelocityField = field;
        mVelocityFunction = nullptr;
        mAdvectionWeight = weight;
    }
    // Velocity as a function of position (pixels per unit time), scaled like the field. Used instead
    // of a sampled field when the flow is known in closed form (e.g. the Enright test), so springls
    // are advected without interpolation error.
    using VelocityFunction = std::function<Vec2(const Vec2 &position)>;
    void setVelocityFunction(VelocityFunction velocity, float weight)
    {
        mVelocityFunction = std::move(velocity);
        mVelocityField.clear();
        mAdvectionWeight = weight;
    }
    void setCurvatureWeight(float w) { mCurvatureWeight = w; }
    void setAdvectionSchedule(AdvectionSchedule schedule) { mSchedule = std::move(schedule); }
    void setMaxIterations(uint64_t n) { mMaxIterations = n; }
    void setResamplingEnabled(bool e) { mResampleEnabled = e; }
    void setTrackingEnabled(bool e) { mTrackingEnabled = e; }

    // Builds the level set and seeds springls from its iso-contour.
    virtual bool init();
    // One time step (up to three advection sub-steps). Returns false when done.
    virtual bool step();

    const Manifold2D &getManifold() const { return mManifold; }
    // Uniform narrow-band level set (empty for adaptive variants).
    const ImageF &getLevelSet() const { return mBand.phi(); }
    const ImageF &getUnsignedLevelSet() const { return mUnsignedLevelSet; }
    double getSimulationTime() const { return mSimulationTime; }
    uint64_t getIteration() const { return mIteration; }
    // Number of advection sub-steps (advect() calls) so far.
    uint64_t getAdvectionSubstepCount() const { return mAdvectionSubstepCount; }
    bool isScheduleComplete() const { return mSchedule.weight && mSimulationTime >= mSchedule.endTime; }
    // Size class of a springl (1 for uniform springls; the cell size where it was created for
    // adaptive springls). Element sizes and resampling distances scale with it.
    float springlScale(size_t springl) const { return (springl < mSpringlScales.size()) ? mSpringlScales[springl] : 1.0f; }

protected:
    std::string mName = "Spring Level Set";
    ImageF mInitialLevelSet;
    ImageV2 mVelocityField;
    VelocityFunction mVelocityFunction;
    float mAdvectionWeight = 1.0f;
    float mCurvatureWeight = 0.0f;
    AdvectionSchedule mSchedule;
    uint64_t mMaxIterations = std::numeric_limits<uint64_t>::max();
    bool mResampleEnabled = true;
    bool mTrackingEnabled = true;
    double mSimulationTime = 0.0;
    uint64_t mIteration = 0;
    uint64_t mAdvectionSubstepCount = 0;

    NarrowBand2D mBand;
    ImageF mUnsignedLevelSet;
    Manifold2D mManifold;
    // Nearest arm vertex of another springl for each arm vertex (-1 for none).
    std::vector<int> mNearestNeighbor;
    // Size classes, parallel to the springls when mScaledSpringls.
    bool mScaledSpringls = false;
    std::vector<float> mSpringlScales;
    SpringlHashGrid2D mSpringlGrid;
    // State captured before resampling, for tracking.
    Points2 mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities;

    // --- Extension points (each keeps every per-springl array in sync). ---
    // Builds the level set from mInitialLevelSet and extracts its contour.
    virtual void initLevelSet();
    virtual void seedSpringls();
    virtual void clearSpringls();
    // Appends one springl with zero velocity and no correspondence; returns its index.
    size_t appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale = 1.0f);
    // Keeps springl i iff keep[i] != 0.
    virtual void compactSpringls(const std::vector<uint8_t> &keep);
    virtual void snapshotForTracking();
    // Whether springl newIndex may inherit tracking from snapshot springl oldIndex.
    virtual bool canTrackFrom(size_t, size_t) const { return true; }
    // Re-solves the iso-contour at the current level set (before filling gaps).
    virtual void solveContourForFill();
    // One level-set update per advection sub-step: unsigned distance, signed evolution, contour.
    virtual void evolveLevelSetAndContour();
    // Whether each sub-step updates the level set right after advection, before relaxation (default:
    // relax first, then update the level set).
    virtual bool rebuildsLevelSetBeforeRelax() const { return false; }
    // Called after each resampling pass and at the end of init().
    virtual void afterResample() { updateUnsignedLevelSet(); }
    // Called after the signed level-set iterations, before the contour is re-solved.
    virtual void afterLevelSetUpdate() {}
    virtual int fill();
    virtual int contract();
    // Size class of contour vertex v (fill distance of springls created there).
    virtual float contourScale(uint32_t) const { return 1.0f; }
    // Neighbor-attraction search scale for an arm vertex at `position` of `springl`.
    virtual float attractionScale(const Vec2 &, size_t springl) const { return springlScale(springl); }
    virtual bool canAttract(size_t, size_t) const { return true; }
    // Whether contour refinement may snap iso-vertexes onto this springl's particle.
    virtual bool refinesContourWith(size_t) const { return true; }

    // --- Building blocks. ---
    void refreshSpringlGrid() { mSpringlGrid.update(mManifold.springlParticles, mManifold.springlVertexes); }
    // maxDistance < 0 uses tuning.unsignedMaxDistance.
    void updateUnsignedLevelSet(float maxDistance = -1.0f);
    void updateSignedLevelSet(float maxStep = 0.5f);
    void solveUniformContour();
    // Smooths the initial contour toward the signed level set's zero crossing.
    void refineInitialContour();
    // Snaps iso-vertexes onto nearby springl particles and evens out their spacing.
    void refineContour(int iterations, float proximity, float stepSize);
    // maxDistance < 0 uses tuning.nearestNeighborDistance.
    void updateNearestNeighbors(float maxDistance = -1.0f);
    void relax(float timeStep);
    void relax();  // finds neighbors, then relaxes
    void relaxWithRememberedNeighbors();
    void relaxSpringl(size_t idx, float timeStep, Vec2 &f1, Vec2 &f2) const;
    // Clears keep[i] for every springl i with include(i) that is too far from the current
    // iso-contour or has degenerate arms; returns how many were cleared.
    int markContractions(std::vector<uint8_t> &keep, const std::function<bool(size_t)> &include) const;
    void updateTracking(float maxDistance);
    void resetTrackingIdentity();
    Vec2 traceInitial(Vec2 pt) const;
    // Moves springls by at most maxStep pixels and maxTimeStep time; returns the time step.
    float advect(float maxStep, float maxTimeStep = std::numeric_limits<float>::infinity());
    // Flow velocity at `position` and time `time`.
    Vec2 particleVelocity(const Vec2 &position, double time) const;
    // Time step of one advection sub-step given the stage-1 particle velocities k1: by default the
    // step that moves the fastest particle maxStep pixels (maxStep when all are slower than 1 pixel
    // per unit time). advect() then caps it at maxTimeStep.
    virtual float advectionTimeStep(const Points2 &k1, float maxStep) const;
    float distanceFieldSpeed(int i, int j) const;
    Vec2 unsignedPull(int i, int j) const;
    Vec2 signedGradientPull(float x, float y) const;
};
}  // namespace springls
