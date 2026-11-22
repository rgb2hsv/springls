#pragma once
#include <core/Image.hpp>
#include <dweave/Manifold2D.hpp>
#include <levelset/LevelSet2D.hpp>
#include <spatial/SpringlHashGrid2D.hpp>

#include <functional>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>

namespace cereal
{
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
}  // namespace cereal

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
    // Region label of the background in multi-object methods (objects are 1..N). The background
    // has no curves and no springls.
    static constexpr int kBackgroundLabel = 0;

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
        // After each level-set evolution, band samples within a cell of a springl, beside it and on
        // the wrong side of it, take its signed distance (particle-sample correction; tightens how
        // closely the level set tracks the springls).
        bool springlCorrection = true;
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
    // Curvature weight of the background region in multi-object methods.
    float backgroundCurvatureWeight = 0.3f;
    // Prints progress to std::cerr every kLogInterval steps.
    bool verbose = false;
    static constexpr uint64_t kLogInterval = 10;

    SpringLevelSet2D();
    virtual ~SpringLevelSet2D() = default;
    const std::string &name() const { return mName; }

    // Signed distance (pixels, negative inside) of the initial interface.
    void setInitialLevelSet(const ImageF &signedDistance) { mInitialLevelSet = signedDistance; }
    // Initial regions of a multi-object method: a label per pixel and the unsigned distance (pixels)
    // to the nearest region boundary. Ignored by single-object methods.
    void setInitialLabels(const ImageI &labels, const ImageF &distance)
    {
        mInitialLabels = labels;
        mInitialLabelDistance = distance;
    }
    // Whether springls and curves carry region labels (multi-object methods).
    bool isLabeled() const { return mLabeled; }
    int springlLabel(size_t springl) const { return mLabeled ? mManifold.springlLabels[springl] : kBackgroundLabel; }
    int curveLabel(size_t curve) const { return mLabeled ? mManifold.curveLabels[curve] : kBackgroundLabel; }
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
    // Checkpoints: the simulation state after init() or any step(), so a run can resume from it, as
    // one cereal portable binary stream (springls, scalars, every level set and grid). Configuration
    // is not stored: set the initial level set, velocity, schedule and tuning the same way, call
    // init(), then loadState(). The state records the method (name()); loadState() refuses another
    // method's state. Return false with a message on failure.
    bool saveState(std::ostream &state, std::string &error) const;
    bool loadState(std::istream &state, std::string &error);
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
    // Multi-object methods set mLabeled in their constructor; springls, curves and resampling then
    // respect region labels (springls interact only with their own label).
    bool mLabeled = false;
    // Labeled contours: 1 for the region junction vertexes of the contour (see FindJunction), as
    // the last extraction reported them (empty: none).
    std::vector<uint8_t> mContourJunctions;
    ImageI mInitialLabels;
    ImageF mInitialLabelDistance;
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

    // The signed level set, evolved toward the springls.
    LevelSet2D mBand;
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
    std::vector<int> mOldLabels;

    // --- Extension points (each keeps every per-springl array in sync). ---
    // Builds the level set from mInitialLevelSet and extracts its contour.
    virtual void initLevelSet();
    virtual void seedSpringls();
    virtual void clearSpringls();
    // Appends one springl with zero velocity and no correspondence; returns its index.
    size_t appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, float scale = 1.0f, int label = kBackgroundLabel);
    // Keeps springl i iff keep[i] != 0.
    virtual void compactSpringls(const std::vector<uint8_t> &keep);
    virtual void snapshotForTracking();
    // Whether springl newIndex may inherit tracking from snapshot springl oldIndex.
    virtual bool canTrackFrom(size_t newIndex, size_t oldIndex) const { return !mLabeled || springlLabel(newIndex) == mOldLabels[oldIndex]; }
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
    virtual bool canAttract(size_t springl, size_t other) const { return !mLabeled || springlLabel(springl) == springlLabel(other); }
    // Whether contour refinement may snap iso-vertexes onto this springl's particle.
    virtual bool refinesContourWith(size_t) const { return true; }

    // Pixel size of the uniform level-set grid (the unsigned field is sampled on it).
    virtual Vec2i levelSetSize() const { return Vec2i(mBand.phi().width(), mBand.phi().height()); }

    // --- Building blocks. ---
    // The particle-sample correction (see Tuning::springlCorrection) from the springls with
    // include(springl) (all if empty); returns how many band pixels changed sign.
    int correctSignsFromSpringls(const std::function<bool(size_t)> &include = nullptr);
    void refreshSpringlGrid() { mSpringlGrid.update(mManifold.springlParticles, mManifold.springlVertexes); }
    // maxDistance < 0 uses tuning.unsignedMaxDistance.
    void updateUnsignedLevelSet(float maxDistance = -1.0f);
    void updateSignedLevelSet(float maxStep = 0.5f);
    void solveUniformContour();
    // Smooths the initial contour toward the signed level set's zero crossing and evens out its
    // vertex spacing.
    void refineInitialContour();
    // Evens out the vertex spacing of the initial contour without changing its shape (labeled
    // contours, whose crossings are exact): springls are seeded one per contour segment. Each curve
    // keeps its vertexes, redistributed at equal arc length along its polyline between the region
    // junction vertexes (mContourJunctions), which stay put.
    void evenInitialContourSpacing();
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
    // Descends |phi0| onto the initial interface (of region `label` in multi-object methods).
    Vec2 traceInitial(Vec2 pt, int label = kBackgroundLabel) const;
    // Signed initial distance of region `label` at p (bilinear; negative inside the region).
    float initialSignedValue(const Vec2 &p, int label) const;
    // Moves springls by at most maxStep pixels and maxTimeStep time; returns the time step.
    float advect(float maxStep, float maxTimeStep = std::numeric_limits<float>::infinity());
    // Flow velocity at `position` and time `time`.
    Vec2 particleVelocity(const Vec2 &position, double time) const;
    // Sub-steps of a method that moves its interface with the flow itself (no springls): up to
    // tuning.advectionSubstepsPerStep times, substep(t, dt) moves it from the current time t for
    // dt, the time in which the fastest interface point (maxSpeed(t), base units per unit time)
    // moves maxStep, capped by the schedule; the clock then advances by dt.
    void advanceWithFlow(float maxStep, const std::function<float(double)> &maxSpeed, const std::function<void(double, float)> &substep);
    // Largest flow speed (base units per unit time) at `points` at time t.
    float maxContourSpeed(const Points2 &points, double t) const;
    // Time step of one advection sub-step given the stage-1 particle velocities k1: by default the
    // step that moves the fastest particle maxStep pixels (maxStep when all are slower than 1 pixel
    // per unit time). advect() then caps it at maxTimeStep.
    virtual float advectionTimeStep(const Points2 &k1, float maxStep) const;
    // dt, shortened until maxSpeed (base units per unit time) sampled over [t0, t0 + dt] (at its middle
    // and end, besides startSpeed at t0) moves at most maxStep: the CFL condition of a time-varying
    // flow (the schedule weight), whose start speed alone does not bound the step.
    static double limitStepOverInterval(double dt, float maxStep, float startSpeed, double t0, const std::function<float(double)> &maxSpeed);
    // Largest particle displacement (base pixels) of one advection sub-step in step().
    virtual float advectionMaxStep() const { return tuning.maxStepPixels; }
    // Checkpoint state of this class and its bases (each override writes its base's first).
    virtual void writeState(cereal::PortableBinaryOutputArchive &ar) const;
    virtual void readState(cereal::PortableBinaryInputArchive &ar);
    Vec2 unsignedPull(int i, int j) const;
    Vec2 signedGradientPull(float x, float y) const;
};
}  // namespace springls
