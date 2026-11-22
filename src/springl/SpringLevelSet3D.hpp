#pragma once
#include <core/Volume.hpp>
#include <levelset/LevelSet3D.hpp>
#include <spatial/SpringlHashGrid3D.hpp>
#include <springl/Manifold3D.hpp>

#include <functional>
#include <iosfwd>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace cereal
{
class PortableBinaryOutputArchive;
class PortableBinaryInputArchive;
}  // namespace cereal

namespace springls
{
// Best similarity transform (rotation R, uniform scale s) taking arms `from` onto arms `to`
// (least squares of |s R from[k] - to[k]|). Returns false when the fit is degenerate.
bool FitSimilarity3D(std::span<const Vec3> from, std::span<const Vec3> to, Eigen::Matrix3f &R, float &s);

// Spring level sets in 3D [Lucas, Kazhdan and Taylor, IEEE TVCG 2012]; 3D counterpart of
// SpringLevelSet2D.
//
// The interface is carried by springls: small surface polygons (a particle with arms to
// 3 to kMaxSpringlSize vertexes: an iso-surface face, see ElementType, or for DoubleWeave a
// dual-mesh face) advected by the flow. A narrow-band level set is evolved toward
// the unsigned distance field of the springls every sub-step; its iso-surface defines the
// topology and is used to resample the springls (contract springls far from it or degenerate,
// fill iso-surface faces no springl covers).
//
// Each advection sub-step:
//   1. Particles and arm vertexes are advected by the flow (RK4 with time-dependent weights).
//   2. Neighbors are found on the advected vertexes and remembered for relaxation.
//   3. Each springl becomes the similarity transform (rotation + uniform scale about its advected
//      particle) of its previous shape that best fits its advected vertexes, so it stays rigid
//      and its orientation follows the flow.
//   4. Relaxation, level-set evolution and iso-surface extraction, then resampling and tracking.
//
// Positions are in level-set voxels. Typical use:
//   sim.setInitialLevelSet(phi); sim.setVelocityField(v, 1); sim.init(); while (sim.step()) {}
class SpringLevelSet3D
{
public:
    static constexpr float kParticleRadius = 0.05f;
    static constexpr float kExtent = 0.5f;  // springl half-size, in voxels
    static constexpr float kMinArmLength = 0.001f;

    // Runtime-tunable parameters (distances in voxels). Defaults were tuned on the 3D Enright
    // test (64^3, T = 3) in wedge.
    struct Tuning {
        // Relaxation.
        float armRestLength = 2.0f * kParticleRadius;           // rest length of each arm spring
        float neighborRestSeparation = 2.0f * kParticleRadius;  // target gap between a vertex and its neighbor edges
        float springConstant = 0.3f;
        float sharpness = 5.0f;  // strength of neighbor attraction
        float nearestNeighborDistance = 0.5f;
        int maxNearestNeighbors = 2;
        int relaxIterations = 4;
        float relaxTimeStep = 0.01f;
        // Resampling (fill / contract).
        float fillDistance = 0.5f;
        float contractDistance = 1.6f;  // max particle distance from the iso-surface
        float maxArmLength = 1.5f;
        float minArea = 0.05f;
        float maxArea = 4.0f;
        float minAspectRatio = 0.1f;
        int maxFillPasses = 3;
        bool fixNormals = true;  // flip springls whose normal opposes the level-set gradient
        // Level set and time stepping.
        int signedEvolveIterations = 16;
        float unsignedMaxDistance = 3.5f;
        float maxStepVoxels = 0.333333f;   // max particle displacement per advection sub-step
        int advectionSubstepsPerStep = 3;  // advection sub-steps (each with a level-set update) per step()
        int refineIterations = 5;
        float refineProximity = 3.0f;
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
    // Time-varying advection: before each sub-step the vector-field weight is weight(t); sub-steps
    // are capped by maxTimeStep and clamped so time never passes endTime.
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

    explicit SpringLevelSet3D(ElementType elementType = ElementType::Quad);
    virtual ~SpringLevelSet3D() = default;
    const std::string &name() const { return mName; }

    // Signed distance (voxels, negative inside) of the initial interface.
    void setInitialLevelSet(const VolumeF &signedDistance) { mInitialLevelSet = signedDistance; }
    // Velocity field in voxels per unit time, scaled by `weight` (or the schedule weight).
    void setVelocityField(const VolumeV3 &field, float weight)
    {
        mVelocityField = field;
        mVelocityFunction = nullptr;
        mAdvectionWeight = weight;
    }
    // Velocity as a function of position (voxels per unit time), scaled like the field. Used instead
    // of a sampled field when the flow is known in closed form (e.g. the Enright test), so springls
    // are advected without interpolation error.
    using VelocityFunction = std::function<Vec3(const Vec3 &position)>;
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
    bool isResamplingEnabled() const { return mResampleEnabled; }
    void setTrackingEnabled(bool e) { mTrackingEnabled = e; }

    // Builds the level set and seeds springls from its iso-surface.
    virtual bool init();
    // After init(): resamples the springls against the fixed level set, repeating relax, contract and fill
    // until the springls stabilize (an iteration removes and adds at most tolerance times their count in all), or
    // maxIterations ran. The springls then start
    // over as the initial interface (zero velocity, correspondence at their particles). Returns the
    // iterations run (0 when maxIterations <= 0).
    int settle(int maxIterations, float tolerance = 1e-4f);
    // One time step (up to three advection sub-steps). Returns false when done.
    virtual bool step();

    // Springl and iso-surface element type: triangles or quads.
    ElementType elementType() const { return mManifold.elementType; }
    int elementSize() const { return mManifold.elementSize(); }
    const Manifold3D &getManifold() const { return mManifold; }

    // Checkpoints: the simulation state after init() or any step(), so a run can resume from it, in
    // two files: `state` (cereal portable binary: the springls, scalars and band bookkeeping) and
    // `levelSets` (an OpenVDB file, written with tinyvdb, holding every level set: "phi", "previous"
    // and "unsigned" volumes of the uniform methods, the adaptive octree's "phi_d<depth>" grids, and
    // each DoubleWeave constellation's own level set prefixed "primal_" / "dual_"). Configuration
    // is not stored: set the initial level set, velocity, schedule and tuning the same way, call
    // init(), then loadState(). Both files record the method (name()) and element type, and
    // loadState() refuses a mismatch or files of different steps. Return false with a message on
    // failure.
    bool saveState(std::ostream &state, std::ostream &levelSets, std::string &error) const;
    bool loadState(std::istream &state, std::istream &levelSets, std::string &error);
    const VolumeF &getLevelSet() const { return mBand.phi(); }
    const VolumeF &getUnsignedLevelSet() const { return mUnsignedLevelSet; }
    double getSimulationTime() const { return mSimulationTime; }
    uint64_t getIteration() const { return mIteration; }
    // Number of advection sub-steps (advect() calls) so far.
    uint64_t getAdvectionSubstepCount() const { return mAdvectionSubstepCount; }
    bool isScheduleComplete() const { return mSchedule.weight && mSimulationTime >= mSchedule.endTime; }
    // Size class of a springl (1 for uniform springls; the cell size where it was created for
    // adaptive springls). Element sizes and resampling distances scale with it.
    float springlScale(size_t springl) const { return (springl < mSpringlScales.size()) ? mSpringlScales[springl] : 1.0f; }

protected:
    std::string mName = "Spring Level Set 3D";
    VolumeF mInitialLevelSet;
    VolumeV3 mVelocityField;
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
    LevelSet3D mBand;
    VolumeF mUnsignedLevelSet;
    Manifold3D mManifold;
    // Up to tuning.maxNearestNeighbors edges of other springls near each arm vertex.
    std::vector<std::vector<SpringlEdge>> mNearestNeighbors;
    SpringlHashGrid3D mSpringlGrid;
    // Size classes, parallel to the springls when mScaledSpringls.
    bool mScaledSpringls = false;
    std::vector<float> mSpringlScales;
    // State captured before resampling, for tracking.
    Points3 mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities;
    std::vector<uint32_t> mOldVertexSpringl;

    // Checkpoint contents: each subclass writes and reads its own state after its base's, its level
    // sets as VDB grids (writeLevelSets / readLevelSets) and the rest with cereal.
    virtual void writeState(cereal::PortableBinaryOutputArchive &ar) const;
    virtual void readState(cereal::PortableBinaryInputArchive &ar);
    virtual bool writeLevelSets(VdbWriter &writer, std::string &error) const;
    virtual bool readLevelSets(const VdbReader &reader, std::string &error);

    // --- Extension points (each keeps every per-springl array in sync). ---
    // Builds the level set from mInitialLevelSet and extracts its iso-surface.
    virtual void initLevelSet();
    virtual void seedSpringls();
    virtual void clearSpringls();
    // Appends one springl (3 to kMaxSpringlSize vertexes) facing along the level-set gradient, with zero velocity and
    // no correspondence; returns its index.
    size_t appendSpringl(const Vec3 &particle, std::span<const Vec3> vertexes, float scale = 1.0f);
    // Keeps springl i iff keep[i] != 0.
    virtual void compactSpringls(const std::vector<uint8_t> &keep);
    virtual void snapshotForTracking();
    // Whether springl newIndex may inherit tracking from snapshot springl oldIndex.
    virtual bool canTrackFrom(size_t, size_t) const { return true; }
    // Whether springl `springl` may be attracted to springl `other`.
    virtual bool canAttract(size_t, size_t) const { return true; }
    // Whether iso-surface refinement may snap vertexes onto this springl's particle.
    virtual bool refinesContourWith(size_t) const { return true; }
    // One level-set update per advection sub-step: unsigned distance, signed evolution, iso-surface.
    virtual void evolveLevelSetAndContour();
    // Whether each sub-step updates the level set right after advection, before relaxation (default:
    // relax first, then update the level set).
    virtual bool rebuildsLevelSetBeforeRelax() const { return false; }
    // Called after the signed level-set iterations, before the iso-surface is re-solved.
    virtual void afterLevelSetUpdate() {}
    virtual int fill();
    virtual int contract();
    // Size class of iso-surface vertex v (springls created there take the smallest of their face's).
    virtual float contourScale(uint32_t) const { return 1.0f; }
    // Neighbor-attraction search scale for an arm vertex at `position` of `springl`.
    virtual float attractionScale(const Vec3 &, size_t springl) const { return springlScale(springl); }
    // Whether markContractions() should also remove springl i (e.g. too small for its grid cell).
    virtual bool isUndersized(size_t) const { return false; }
    // Unit gradient of the signed level set at a point (zero where flat).
    virtual Vec3 levelSetGradient(const Vec3 &p) const;

    // --- Building blocks. ---
    // The particle-sample correction (see Tuning::springlCorrection) from the springls with
    // include(springl) (all if empty); returns how many band voxels changed sign.
    int correctSignsFromSpringls(const std::function<bool(size_t)> &include = nullptr);
    void refreshSpringlGrid() { mSpringlGrid.update(mManifold.springlParticles, mManifold.springlVertexes, mManifold.springlStart); }
    // Unsigned distance to the springls with include(springl), near the current level set's
    // interface (maxDistance elsewhere). maxDistance < 0 uses tuning.unsignedMaxDistance.
    void updateUnsignedLevelSet(float maxDistance = -1.0f, const std::function<bool(size_t)> &include = {});
    void updateSignedLevelSet(float maxStep = 0.5f);
    // Extracts and refines the iso-surface of the current level set into mesh.
    void solveContour(SurfaceMesh3D &mesh) const;
    void solveContour() { solveContour(mManifold.isoSurface); }
    // Snaps iso-surface vertexes onto nearby springl particles and evens out their spacing.
    void refineContour(SurfaceMesh3D &mesh, int iterations, float proximity, float stepSize) const;
    // Points iso-surface vertex normals outward (along the level-set gradient).
    void orientVertexNormals(SurfaceMesh3D &mesh) const;
    void updateNearestNeighbors(float maxDistance = -1.0f);
    void relax(float timeStep);
    void relax();  // finds neighbors, then relaxes
    void relaxWithRememberedNeighbors();
    void relaxSpringl(size_t idx, float timeStep, Points3 &updates) const;
    // Clears keep[i] for every springl i with include(i) that is too far from the current
    // iso-surface or degenerate; returns how many were cleared.
    int markContractions(std::vector<uint8_t> &keep, const std::function<bool(size_t)> &include) const;
    // Flips springls whose normal opposes the level-set gradient.
    void fixNormals();
    void updateTracking(float maxDistance);
    void resetTrackingIdentity();
    Vec3 traceInitial(Vec3 pt) const;
    // Moves springls by at most maxStep voxels and maxTimeStep time; returns the time step.
    float advect(float maxStep, float maxTimeStep = std::numeric_limits<float>::infinity());
    // Flow velocity at `position` and time `time`.
    Vec3 particleVelocity(const Vec3 &position, double time) const;
    // Sub-steps of a method that moves its interface with the flow itself (no springls): up to
    // tuning.advectionSubstepsPerStep times, substep(t, dt) moves it from the current time t for
    // dt, the time in which the fastest interface point (maxSpeed(t), base units per unit time)
    // moves maxStep, capped by the schedule; the clock then advances by dt.
    void advanceWithFlow(float maxStep, const std::function<float(double)> &maxSpeed, const std::function<void(double, float)> &substep);
    // Largest flow speed (base units per unit time) at `points` at time t.
    float maxContourSpeed(const Points3 &points, double t) const;
    // Time step of one advection sub-step given the stage-1 particle velocities k1: by default the
    // step that moves the fastest particle maxStep voxels (maxStep when all are slower than 1 voxel
    // per unit time). advect() then caps it at maxTimeStep.
    virtual float advectionTimeStep(const Points3 &k1, float maxStep) const;
    Vec3 unsignedPull(int i, int j, int k) const;
};
}  // namespace springls
