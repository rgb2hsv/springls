#include <lod/MeshTracking3D.hpp>

#include <core/Common.hpp>
#include <core/Profile.hpp>
#include <lod/CvtCharts.hpp>
#include <mesh/Icosphere.hpp>
#include <springl/SpringlSignField3D.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>

namespace springls
{
namespace
{
// |phi| on the finest leaves is the true distance out to this many finest cells, then clamped (as for the
// semi-Lagrangian springls).
constexpr float kDistanceCells = 4.0f;
}  // namespace

MeshTracking3D::MeshTracking3D(ElementType) : AdaptiveSpringLevelSet3D(ElementType::Triangle)
{
    mName = "Mesh Tracking 3D";
    mSplitOnRefine = false;  // no springls
}

void MeshTracking3D::setInitialSphere(const Vec3 &center, float radius)
{
    mSphereCenter = center;
    mSphereRadius = radius;
    mHasSphere = true;
}

bool MeshTracking3D::init()
{
    if (mInitialLevelSet.empty() || !mHasSphere) return false;
    mSimulationTime = 0.0;
    mIteration = 0;
    mAdvectionSubstepCount = 0;
    mUnsignedLevelSet.clear();
    clearSpringls();
    mMesh = TriangleSurface();
    const LoopMesh sphere = MakeIcosphere(meshTuning.sphereLevels, mSphereRadius, mSphereCenter.cast<double>());
    mMesh.vertexes = sphere.vertices;
    mMesh.triangles = sphere.triangles;
    mMesh.faceRegion.assign(mMesh.triangles.size(), 0);
    // The charts, once: the meshlet pipeline's base level of the initial mesh (or the raw CVT). They are the
    // mesh's regions from then on, and remeshing adds new triangles to them.
    mHasCharts = false;
    if (meshTuning.chartTriangles > 0) {
        CvtChartParams chartParams;
        chartParams.meshletBaseLevel = meshTuning.meshletCharts;
        chartParams.maxTrianglesPerChart = meshTuning.chartTriangles;
        chartParams.chartFill = meshTuning.chartFill;
        chartParams.chartCount = std::max(1, static_cast<int>(mMesh.triangles.size()) / meshTuning.chartTriangles);
        std::vector<int> charts = SegmentCvtCharts(mMesh.vertexes, mMesh.triangles, chartParams);
        if (charts.size() == mMesh.triangles.size()) {
            mMesh.faceRegion = std::move(charts);
            mHasCharts = true;
        }
    }
    std::vector<float> lengths;
    for (const auto &t : mMesh.triangles) lengths.push_back((mMesh.vertexes[t[1]] - mMesh.vertexes[t[0]]).norm());
    std::nth_element(lengths.begin(), lengths.begin() + lengths.size() / 2, lengths.end());
    mTargetEdge = lengths.empty() ? 1.0f : lengths[lengths.size() / 2];
    mCheck.reset();
    mMaxFolded = mMaxOverlapping = 0;
    initLevelSet();  // from the initial level set; from the first sub-step on it is rebuilt from the mesh
    return true;
}

void MeshTracking3D::advectMesh(double t, float dt)
{
    ProfileScope profileScope("mesh tracking: advect");
    const double tMid = t + 0.5 * dt, tEnd = t + dt;
    Points3 &points = mMesh.vertexes;
    const int n = static_cast<int>(points.size());
    SPRINGLS_OMP(omp parallel for if (n >= kParallelMinLightItems))
    for (int i = 0; i < n; i++) {
        const Vec3 p = points[i];
        const Vec3 k1 = particleVelocity(p, t);
        const Vec3 k2 = particleVelocity(p + 0.5f * dt * k1, tMid);
        const Vec3 k3 = particleVelocity(p + 0.5f * dt * k2, tMid);
        const Vec3 k4 = particleVelocity(p + dt * k3, tEnd);
        points[i] = p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    }
}

void MeshTracking3D::rebuildLevelSet()
{
    ProfileScope profileScope("mesh tracking: rebuild level set");
    adaptToPoints(mMesh.vertexes);  // the band around the mesh as it is now
    // Each triangle an element of three vertexes about its centroid: the fan of its corners about the centroid is
    // the triangle itself.
    const size_t F = mMesh.triangles.size();
    Points3 centroids, corners;
    std::vector<uint32_t> start;
    centroids.reserve(F);
    corners.reserve(3 * F);
    start.reserve(F + 1);
    for (const auto &t : mMesh.triangles) {
        start.push_back(static_cast<uint32_t>(corners.size()));
        Vec3 centroid = Vec3::Zero();
        for (uint32_t v : t) {
            corners.push_back(mMesh.vertexes[v]);
            centroid += mMesh.vertexes[v];
        }
        centroids.push_back(centroid / 3.0f);
    }
    start.push_back(static_cast<uint32_t>(corners.size()));
    SignFieldInput3D input;
    input.grid = &mGrid;
    input.particles = &centroids;
    input.vertexes = &corners;
    input.start = &start;
    input.springls.resize(F);
    std::iota(input.springls.begin(), input.springls.end(), 0u);
    float finestCell = std::numeric_limits<float>::infinity();
    for (int leaf : mGrid.leaves()) finestCell = std::min(finestCell, mGrid.blocks()[leaf].h);
    SignFieldParams3D params;
    params.capsuleCells = meshTuning.capsuleCells;
    params.distanceClamp = std::min(adaptiveTuning.bandWidth, kDistanceCells * (std::isfinite(finestCell) ? finestCell : 1.0f));
    SpringlSignField3D field;
    field.build(input, params);
    FillGridFromSignField(mGrid, field, adaptiveTuning.bandWidth, params.distanceClamp);
    mGrid.synchronize();
    mGrid.updateActive();
    extractGridSurface();
}

bool MeshTracking3D::step()
{
    // Each sub-step (the fastest mesh vertex moves at most maxStepVoxels base voxels, as springls do): the mesh moves
    // with the flow, then the level set is rebuilt from it.
    advanceWithFlow(
        tuning.maxStepVoxels, [this](double t) { return maxContourSpeed(mMesh.vertexes, t); },
        [this](double t, float dt) {
            advectMesh(t, dt);
            rebuildLevelSet();
        });
    RemeshParams params;
    params.targetEdge = mTargetEdge;
    params.splitRatio = meshTuning.splitRatio;
    params.collapseRatio = meshTuning.collapseRatio;
    params.minAngleDegrees = meshTuning.minAngleDegrees;
    params.passes = meshTuning.remeshPasses;
    {
        ProfileScope profileScope("mesh tracking: remesh");
        if (mHasCharts) {
            RemeshSurfaceByCharts(mMesh, params);  // the charts (regions) keep their triangles and borders
        } else {
            RemeshSurface(mMesh, params);
        }
    }
    // Vertexes no triangle uses (collapsed) are dropped from time to time.
    // A closed mesh has about half as many vertexes as triangles.
    if (mMesh.vertexes.size() > mMesh.triangles.size() + 1024) {
        std::vector<int> map(mMesh.vertexes.size(), -1);
        Points3 kept;
        for (auto &t : mMesh.triangles) {
            for (uint32_t &v : t) {
                if (map[v] < 0) {
                    map[v] = static_cast<int>(kept.size());
                    kept.push_back(mMesh.vertexes[v]);
                }
                v = static_cast<uint32_t>(map[v]);
            }
        }
        mMesh.vertexes = std::move(kept);
    }
    if (meshTuning.checkMesh) {
        ProfileScope profileScope("mesh tracking: check mesh");
        MeshCheckParams checkParams;
        checkParams.foldDegrees = meshTuning.foldDegrees;
        mCheck = std::make_shared<const MeshCheck>(CheckMesh(mMesh, checkParams));
        mMaxFolded = std::max(mMaxFolded, mCheck->foldedEdges);
        mMaxOverlapping = std::max(mMaxOverlapping, mCheck->overlappingPairs);
    } else {
        mCheck.reset();
    }
    mIteration++;
    if (verbose && (mIteration % kLogInterval) == 0) {
        std::cerr << mName << ": step " << mIteration << " t = " << mSimulationTime << ", " << mMesh.triangles.size() << " triangles";
        if (mCheck) std::cerr << ", " << mCheck->foldedEdges << " fold-overs, " << mCheck->overlappingPairs << " overlaps, " << mCheck->nonManifoldEdges << " non-manifold edges";
        std::cerr << std::endl;
    }
    return mIteration < mMaxIterations && !isScheduleComplete();
}
}  // namespace springls
