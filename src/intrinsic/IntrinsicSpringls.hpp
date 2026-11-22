#pragma once
// Spring level sets on a triangle-mesh surface ("intrinsic springls"), one region: springls are short
// segments on an intrinsic Delaunay triangulation (particle and two arm vertexes as surface points),
// and the region's level set is rebuilt from the springls after every advection sub-step
// (ReconstructSignField: semi-Lagrangian, as SpringlSignField2D in 2D), never evolved by a PDE. The
// simulation owns a copy of the triangulation and refines it adaptively where the springls are (Loop
// 1:4 splits at intrinsic edge midpoints, no smoothing: the surface never changes), up to maxLevel
// levels below the input mesh; every surface point it holds (and the owner's tracked points) is
// relocated exactly through each refinement. Behind the band the refinement is undone: when enough of
// the refined area lies beyond the coarsening distance, the refined mesh is rebuilt from the input mesh
// by the same refinement around the current springls (coarsening; every point relocated exactly). Every
// motion and measurement along the surface is geodesic (exponential map, local charts, fast marching).
// The step order and tuning are those of the 2D springls (SpringLevelSet2D), with every length
// constant in units of h_t, the target edge length near the springls: the input length scale (mean
// edge length) over 2^maxLevel.
#include <intrinsic/IntrinsicContour.hpp>
#include <intrinsic/IntrinsicTrace.hpp>
#include <springl/SpringLevelSet2D.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace springls
{
struct IntrinsicSpringl {
    SurfacePoint particle, v0, v1;
};

class IntrinsicSpringls
{
public:
    static constexpr double kBandWidth = 3.5;       // |phi| clamp, in h_t (as MultiNarrowBand2D::kMaxDistance)
    static constexpr double kCapsule = 1.5;         // capsule radius of the reconstruction, in h_t (SpringlSignField2D's capsuleCells)
    static constexpr double kRefineMargin = 2.0;    // faces within kCapsule + kRefineMargin (h_t) of the springls are refined
    static constexpr double kSideCorrection = 1.0;  // vertices this near (h_t) a springl take its side (the 2D springl correction's reach)
    static constexpr int kRegionLabel = 1;          // the contour's label
    static constexpr int kDefaultMaxLevel = 3;
    // The refined mesh is rebuilt when the refined faces (faceLevel > 0) all of whose corners lie beyond
    // coarseningDistance() cover more than this fraction of the refined faces' area.
    static constexpr double kCoarsenFraction = 0.25;
    static constexpr double kCoarsenReach = 1.6;  // in longest input edges, beyond the band (coarseningDistance)

    // The fastest particle moves at most kDefaultMaxStep h_t per advection sub-step (tuning.maxStepPixels).
    // The 2D methods' ⅓ is the upwind level-set PDE's CFL limit; the rebuild here is semi-Lagrangian, so
    // only the springls' own motion limits the step: a step of 1 h_t takes a third of the sub-steps of ⅓
    // (about 2.4× faster over the sphere and stone loops) and keeps the sphere's shape error under 0.1 h_t.
    static constexpr float kDefaultMaxStep = 1.0f;

    SpringLevelSet2D::Tuning tuning = DefaultTuning();  // lengths in units of h_t
    bool verbose = false;
    // Velocity at a surface point of triangulation() (the simulation's refined copy) and time, in the
    // point's face layout (length per unit time).
    using Velocity = std::function<Vec2d(const SurfacePoint &, double t)>;
    // As SpringLevelSet2D::AdvectionSchedule: the velocity is scaled by weight(t); sub-steps are capped by
    // maxTimeStep and never pass endTime.
    struct Schedule {
        std::function<float(double)> weight;
        double endTime = std::numeric_limits<double>::infinity();
        double maxTimeStep = std::numeric_limits<double>::infinity();
    };

    // Copies the triangulation: the simulation refines its copy.
    explicit IntrinsicSpringls(const IntrinsicTriangulation &tri);
    // SpringLevelSet2D's tuning with maxStepPixels = kDefaultMaxStep.
    static SpringLevelSet2D::Tuning DefaultTuning()
    {
        SpringLevelSet2D::Tuning t;
        t.maxStepPixels = kDefaultMaxStep;
        return t;
    }
    // Initial level set per vertex of triangulation() (the input triangulation before the first init):
    // negative inside, |phi| the distance to the region's boundary (at least near it). Vertices inserted
    // by the initial refinement take it interpolated on the input mesh face they lie on.
    void setInitialRegion(const std::vector<double> &phi);
    void setVelocity(Velocity velocity, Schedule schedule);
    void setMaxIterations(uint64_t n) { mMaxIterations = n; }
    // The input length scale (0, the default: the input triangulation's mean edge length); h_t is it over
    // 2^maxLevel.
    void setLengthScale(double h) { mLengthScale = h; }
    // Refinement levels below the input mesh (default 3; 0: no refinement).
    void setMaxLevel(int levels) { mMaxLevel = std::max(0, levels); }
    // Extra surface points (on triangulation()) the owner wants relocated through every refinement, e.g.
    // a Lagrangian reference. Called at each refinement; the pointers must stay valid during it.
    void setTrackedPoints(std::function<std::vector<SurfacePoint *>()> points) { mTrackedPoints = std::move(points); }
    // Coarsening (default on): at the start of each sub-step's refinement, rebuild the refined mesh when
    // enough refined area lies beyond the coarsening distance (see kCoarsenFraction).
    void setCoarsening(bool on) { mCoarsening = on; }
    bool coarsening() const { return mCoarsening; }
    // Rebuilds since init (automatic and rebuildMesh).
    size_t rebuildCount() const { return mRebuilds; }
    // Rebuilds the refined mesh now (as the automatic trigger does): every springl point and tracked point
    // mapped to the input mesh (same 3D position), the input mesh restored (and flipped to intrinsic
    // Delaunay, as the simulation's own copy started), the level set kept on the input vertices, then the
    // refinement passes around the springls (new vertices take phi from their neighbors), arms re-expressed
    // in their particles' new faces, charts dropped and the contour re-extracted.
    void rebuildMesh();
    // Distance from the springls beyond which a refined face counts as stale: the refinement band
    // ((kCapsule + kRefineMargin) h_t) plus (1 + the corner reach) longest input edges L. An input face is
    // split when its nearest corner is within the band plus the corner reach (0.6) of its longest edge, so
    // every face refined from it lies within band + 1.6 L: beyond this distance the one-pass refinement
    // leaves the input mesh, and a refined face there is left behind by the band (the hysteresis is about
    // one input edge beyond where refinement starts).
    double coarseningDistance() const { return (kCapsule + kRefineMargin) * mH + kCoarsenReach * mInputLongest; }
    // Refines around the initial region's boundary, then seeds one springl per contour segment at least an
    // arm's rest length long (a closed contour of only shorter segments gets none). Calling it
    // again restarts from a new initial region on the triangulation as refined so far.
    bool init(std::string &error);
    // One time step (advectionSubstepsPerStep sub-steps). Returns false when done.
    bool step();
    // Removes the springls with keep[s] == 0 (for tests and tools: e.g. opening a gap).
    void removeSpringls(const std::vector<uint8_t> &keep) { compact(keep); }

    const IntrinsicTriangulation &triangulation() const { return mTri; }
    const std::vector<IntrinsicSpringl> &springls() const { return mSpringls; }
    // Arm vectors of each springl in its particle's face layout (v_e = Trace(particle, arm_e)).
    const std::vector<std::array<Vec2d, 2>> &arms() const { return mArms; }
    const std::vector<IntrinsicCurve> &contour() const { return mContour; }
    // The level set per vertex of triangulation(): negative inside, clamped at kBandWidth h_t.
    const std::vector<double> &phi() const { return mPhi; }
    // Nearest arm vertex (2 * springl + arm) of another springl for each arm vertex, or -1.
    const std::vector<int> &nearestNeighbors() const { return mNearest; }
    double time() const { return mTime; }
    uint64_t iteration() const { return mIteration; }
    double meanEdgeLength() const { return mH; }  // h_t, the length unit in use
    int maxLevel() const { return mMaxLevel; }
    // Totals since init: springls contracted and filled, and sub-steps whose reconstruction leaked (no
    // inside: the previous level set was kept).
    size_t contractedTotal() const { return mContractedTotal; }
    size_t filledTotal() const { return mFilledTotal; }
    size_t leakCount() const { return mLeaks; }
    // Every fill and contraction since init, for diagnostics.
    struct ResampleEvent {
        enum class Kind {
            Fill,              // a contour segment's midpoint had no springl within fillDistance
            ContractFar,       // the particle was farther than contractDistance from the contour
            ContractShortArm,  // an arm shorter than kMinEdgeLength
            ContractLongArm    // an arm longer than maxArmLength
        };
        Kind kind = Kind::Fill;
        double time = 0.0;
        Vec3d position = Vec3d::Zero();  // 3D (toExtrinsic): the fill segment's midpoint, the contracted particle
        // In h_t. Fill: the midpoint's distance to the nearest springl (the trigger). Contraction: the
        // particle's distance to the contour (infinite when no contour segment is in its chart).
        double distance = 0.0;
        // In h_t. Fill: how far apart the springls are there (the largest distance from an arm vertex near the
        // midpoint, within distance + fillDistance, to the nearest arm vertex of another springl; at a hole
        // the chain ends have none near). Contraction: the longer arm.
        double separation = 0.0;
        // In h_t. Fill: the length of the contour segment the springl was seeded on. Contraction: 0.
        double length = 0.0;
    };
    const std::vector<ResampleEvent> &resampleEvents() const { return mEvents; }
    // Local charts held in the caches (diagnostics: they are kept only near the springls).
    size_t chartCacheSize() const { return mCharts.size() + mSeedCharts.size(); }
    bool isScheduleComplete() const { return mSchedule.weight && mTime >= mSchedule.endTime; }

private:
    IntrinsicTriangulation mTri;
    double mH = 1.0;
    double mInputLongest = 0.0;  // longest input edge
    double mLengthScale = 0.0;
    int mMaxLevel = kDefaultMaxLevel;
    std::vector<double> mInitialPhi;
    Velocity mVelocity;
    Schedule mSchedule;
    std::function<std::vector<SurfacePoint *>()> mTrackedPoints;
    uint64_t mMaxIterations = std::numeric_limits<uint64_t>::max();
    double mTime = 0.0;
    uint64_t mIteration = 0;
    size_t mContractedTotal = 0, mFilledTotal = 0, mLeaks = 0;
    std::vector<ResampleEvent> mEvents;
    bool mCoarsening = true;
    size_t mRebuilds = 0;
    // U, the distance to the springls, as the last refineNearSpringls returned it (on the current mesh:
    // nothing else changes the mesh), for the coarsening trigger.
    std::vector<double> mDistance;

    std::vector<double> mPhi;
    std::vector<IntrinsicCurve> mContour;
    std::vector<IntrinsicSpringl> mSpringls;
    std::vector<std::array<Vec2d, 2>> mArms;
    std::vector<int> mNearest;
    // Springls by the face of their particle, rebuilt on demand.
    std::unordered_map<uint32_t, std::vector<uint32_t>> mByFace;
    // Local charts, built lazily per face and dropped when a refinement changes their faces. Inside a parallel
    // region chart() only reads mCharts: a missing chart goes to a per-thread cache, valid while mChartEpoch
    // (a process-wide unique number, renewed whenever charts are dropped as stale) is unchanged.
    // ensureCharts builds missing charts in parallel ahead of a parallel loop.
    mutable std::unordered_map<uint32_t, LocalChart> mCharts;
    uint64_t mChartEpoch = 0;
    // The reconstruction's seed charts (depth chosen by SpringlDistance), dropped as mCharts.
    std::unordered_map<uint32_t, std::pair<int, LocalChart>> mSeedCharts;

    const LocalChart &chart(uint32_t face) const;
    void ensureCharts(std::vector<uint32_t> faces);
    void ensureSeedCharts(std::vector<uint32_t> faces, int depth);
    void invalidateCharts();
    // Displacement from `origin` to `target` in origin's face layout, through origin's chart.
    bool relative(const SurfacePoint &origin, const SurfacePoint &target, Vec2d &out) const;
    Vec2d velocityAt(const SurfacePoint &p, double t) const;
    // Springl s's chord and polyline in the chart of `face` (false if its particle is outside it).
    bool springlInChart(const LocalChart &c, size_t s, Vec2d &a0, Vec2d &p, Vec2d &a1) const;
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &byFace();
    std::vector<std::pair<SurfacePoint, SurfacePoint>> springlSegments() const;

    // Refinement: faces longer than h_t (longest edge > 1.01 h_t) below maxLevel whose distance
    // (lower bound from the vertex distances `distance`) is within `band`.
    std::vector<uint8_t> refinementMarks(const std::vector<double> &distance, double band) const;
    // Every springl point (particle, v0, v1) and tracked point.
    std::vector<SurfacePoint *> heldPoints();
    // Springl s's arms re-expressed in its particle's face layout (after its face changed).
    void reexpressArms(size_t s);
    // Splits the marked faces, relocating every held and tracked point; returns the vertex count before.
    size_t refineFaces(const std::vector<uint8_t> &marks);
    void refineInitialRegion();
    // Refines near the springls (up to maxLevel + 1 passes), first rebuilding the mesh when coarsening
    // pays (coarseningDue); returns U, the distance to the springls on the final mesh (also kept).
    std::vector<double> refineNearSpringls();
    std::vector<double> refinementPasses();
    // The stale-area test on the kept U.
    bool coarseningDue() const;
    // The rebuild; returns U on the rebuilt mesh.
    std::vector<double> rebuild();
    // New vertices (index >= from) of mPhi: the mean of their already-valued neighbors.
    void extendPhi(size_t from);
    void reconstructPhi(const std::vector<double> &distance);
    // Drops the cached charts of faces farther than the refinement band from the springls (`distance`: U).
    void pruneCharts(const std::vector<double> &distance);
    // Side of vertex v of the nearest springl within reach (SpringlSide; mByFace current).
    int springlSide(uint32_t v, double reach) const;

    void extractContour();
    // Every arm's crossings of mesh edges (the trace from the particle along the arm).
    std::vector<EdgeCrossing> armCrossings() const;
    // The 1D refinement toward the particles, then every crossing on an edge an arm (else a neighbor bridge)
    // crosses placed on it.
    void refineContour();
    void seedSpringls();
    size_t appendSpringl(const SurfacePoint &particle, const SurfacePoint &v0, const SurfacePoint &v1);
    void setArms(size_t s, const Vec2d &arm0, const Vec2d &arm1);
    void compact(const std::vector<uint8_t> &keep);

    double advect(double maxStep, double maxTimeStep);
    void updateNearestNeighbors();
    void relax(double timeStep);
    void relaxWithRememberedNeighbors();
    void relaxAll();
    int contract();
    int fill();
    // How far apart the springls (index < existing, bucketed in mByFace) are near x: the largest distance from
    // an arm vertex within `reach` of x to the nearest arm vertex of another springl in the chart.
    double springlHole(const LocalChart &c, const Vec2d &x, double reach, size_t existing) const;
};
}  // namespace springls
