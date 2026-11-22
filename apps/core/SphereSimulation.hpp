#pragma once
// The sphere scenario: multi-object springls travelling a geodesic loop on a triangle-mesh surface and back
// (springls::intrinsic::Session). No GL: init() and step() may run on a worker thread; call snapshot() from
// the thread that steps (or while no step runs): it also fills the arrow-sample cache, so it is not const and
// not safe to call concurrently. A window takes it on its worker thread and hands the copy to the UI thread.
#include "Simulation.hpp"

#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace toybox
{
using Vec3f = springls::Vec3;  // single precision, as the renderers want
// A velocity sample: the arrow from `from` to `to` (3D, on the input mesh), the speed there, and the input
// mesh's unit normal at `from` (its area-weighted vertex normals interpolated there).
struct SphereArrow {
    Vec3f from, to;
    Vec3f normal = Vec3f::UnitZ();
    float speed = 0;
};

// Everything a renderer needs, copied out of the live simulation so it can be drawn while the next step runs.
struct SphereSnapshot {
    uint64_t stamp = 0;           // new on every init / step, unique across simulations
    std::vector<Vec3f> vertices;  // the refined triangulation's 3D positions (input and inserted vertices)
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<int> faceLevels;                    // per triangle: its subdivision level below the input mesh (IntrinsicTriangulation::faceLevel)
    std::vector<int> labels;                        // per vertex: the region's label where phi < 0, else 0
    std::vector<float> distances;                   // per vertex: |phi|, the distance to the region's boundary (clamped)
    std::vector<std::pair<Vec3f, Vec3f>> springls;  // v0, v1 of each springl
    std::vector<Vec3f> particles;                   // each springl's particle (its arms run v0 -> particle -> v1)
    // Each springl's unit normal at its particle, outward (the region is on the springls' left: v0 -> v1 with
    // the surface normal up), tangent to the surface: (v1 - v0) x the input face's normal, normalized.
    std::vector<Vec3f> normals;
    std::vector<int> springlLabels;
    std::vector<std::pair<Vec3f, Vec3f>> contour;  // one pair per contour segment
    std::vector<int> contourLabels;
    std::vector<SphereArrow> arrows;
    std::vector<std::vector<Vec3f>> reference;  // the Lagrangian reference's polylines (closed on a closed mesh: the last point joins the first)
    // The area-weighted centroid of the object's regions on the surface (see ObjectFrame) and the unit
    // surface normal there; the scenario's object center when the regions have no area.
    Vec3f objectCenter = Vec3f::Zero(), objectNormal = Vec3f::UnitZ();
    float objectRadius = 0, lengthScale = 0;
    float bandWidth = 0;  // phi's clamp (IntrinsicSpringls::kBandWidth length scales): distances are clamped there
    double time = 0, period = 0;
    uint64_t steps = 0;
    size_t contractedTotal = 0, filledTotal = 0;  // springls contracted / filled since init
    bool complete = false;
    springls::LoopScenario::Metrics metrics;
};

class SphereSimulation : public Simulation
{
public:
    SphereSimulation(const Entry &entry, const SphereOptions &options);

    const Entry &entry() const override { return mEntry; }
    bool init(std::string &error) override;
    bool step() override;
    bool complete() const override { return mSession.complete(); }
    double time() const override;
    uint64_t iteration() const override;
    uint64_t substeps() const override { return 0; }
    size_t springlCount() const override;
    bool supportsCsv() const override { return true; }
    void csvHeader(std::ostream &out) const override;
    void csvRow(std::ostream &out, double stepSeconds) const override;
    void summary(std::ostream &out, double seconds) const override;

    // Arrow samples are spaced `arrowSpacing` length scales apart; each arrow is arrowLength length scales long
    // for the fastest sample.
    SphereSnapshot snapshot(double arrowSpacing = 2.0, double arrowLength = 0.5);
    const springls::intrinsic::Session &session() const { return mSession; }
    const SphereOptions &options() const { return mOptions; }

    // Headless statistics: the initialization line and one line per step.
    void printInit(std::ostream &out) const;
    void printStep(std::ostream &out) const;

private:
    const Entry &mEntry;
    SphereOptions mOptions;
    springls::intrinsic::Session mSession;
    springls::LoopScenario::Metrics mMetrics;
    uint64_t mSteps = 0, mStamp = 0;
    double mSeconds = 0;
    double mSampleSpacing = 0;  // the cached samples' spacing
    std::vector<springls::SurfacePoint> mSamples;
};

// Where the object is: the area-weighted centroid of the parts of its regions (every object label, the same
// per-face clipping as the area metric) on the input mesh, and the area-weighted normal of those parts. The
// centroid of a curved region lies below the surface: it is moved along that normal onto the level of the
// nearest part's centroid. When the normals cancel (a region wrapping the surface) the nearest part's centroid
// and its face normal are used. False (and nothing set) when the regions have no area.
bool ObjectFrame(const springls::intrinsic::Session &session, Vec3f &center, Vec3f &normal);

// Sample points (fixed per session: spaced `spacing` length scales apart over the whole mesh) and their
// arrows: from the point along the geodesic of arrowLength * lengthScale * v / max|v|
// at the current time, signed by the schedule.
std::vector<SphereArrow> SampleArrows(const springls::intrinsic::Session &session, double spacing, double arrowLength);
}  // namespace toybox
