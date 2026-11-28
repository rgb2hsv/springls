#pragma once
#include <lod/AdaptiveRemesh3D.hpp>
#include <lod/MeshCheck3D.hpp>
#include <lod/TriangleSurface3D.hpp>
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <memory>

namespace springls
{
// Mesh tracking with adaptive remeshing (mesh-tracking), without springls and without topology changes: a closed
// triangle mesh is the interface, and its topology never changes. A level set on the adaptive octree (32^3 to
// 256^3 at the defaults) is rebuilt from it after every sub-step by the semi-Lagrangian method of
// SemiLagrangianDweaveLevelSet3D, the mesh's triangles taking the springls' place (SpringlSignField3D: the
// unsigned distance to the triangles, signed by an exterior flood; no level set equation, no CFL limit).
//
// Each sub-step (the fastest mesh vertex moves at most tuning.maxStepVoxels base voxels) the mesh's vertexes move
// with the flow (RK4, the exact velocity), the band is refitted around the mesh, the level set rebuilt and the
// iso-surface extracted. Each step the mesh is then remeshed for quality (AdaptiveRemesh3D, chart by chart): edges
// longer than splitRatio times its median edge at init are split, shorter than collapseRatio times it (or a
// sliver's shortest edge) collapsed, and edges flipped where that raises the smallest angle of their two triangles
// (the local Delaunay criterion). Where sheets of the mesh come together the mesh is left as it is: the level set
// shows their union. Checkpoints are not supported.
class MeshTracking3D : public AdaptiveSpringLevelSet3D
{
public:
    struct MeshTuning {
        int sphereLevels = 5;  // icosphere subdivisions of the initial mesh (20 * 4^n triangles)
        float splitRatio = 1.5f;
        float collapseRatio = 0.5f;
        float minAngleDegrees = 15.0f;
        int remeshPasses = 4;
        // > 0: remesh chart by chart. The charts are made once, from the initial mesh, by meshlet-optimizer's
        // meshlet pipeline run to its base level (CvtCharts.hpp), at most chartTriangles triangles each (its
        // default, 256); meshletCharts false: the raw CVT, chartTriangles triangles per chart on average. They are
        // the mesh's regions: new triangles join the chart they were made from (RemeshSurfaceByCharts).
        // 0: remesh by independent sets over the whole mesh.
        int chartTriangles = 256;
        double chartFill = 0.5;  // share of chartTriangles each chart aims at (sets the chart count; meshlets: 0.375)
        bool meshletCharts = true;
        float capsuleCells = 1.5f;  // capsule radius of the level set rebuild, in finest cells (SpringlSignField3D)
        // Check the mesh after every step for fold-overs (adjacent normals more than foldDegrees apart) and
        // overlaps (triangles sharing no vertex that intersect); see meshCheck().
        bool checkMesh = false;
        float foldDegrees = 120.0f;
    };
    MeshTuning meshTuning;

    explicit MeshTracking3D(ElementType elementType = ElementType::Triangle);
    void setInitialSphere(const Vec3 &center, float radius);
    bool init() override;
    bool step() override;

    // The tracked mesh, in base voxels (vertexes no triangle uses may remain).
    const TriangleSurface &mesh() const { return mMesh; }
    size_t triangleCount() const { return mMesh.triangles.size(); }
    float targetEdge() const { return mTargetEdge; }
    // The chart of every triangle (its region; all 0 without charts).
    const std::vector<int> &charts() const { return mMesh.faceRegion; }
    bool hasCharts() const { return mHasCharts; }
    // The last step's mesh check (null unless meshTuning.checkMesh), and the most seen in any step.
    std::shared_ptr<const MeshCheck> meshCheck() const { return mCheck; }
    int maxFoldedEdges() const { return mMaxFolded; }
    int maxOverlappingPairs() const { return mMaxOverlapping; }

protected:
    TriangleSurface mMesh;
    float mTargetEdge = 0.0f;
    std::shared_ptr<const MeshCheck> mCheck;
    bool mHasCharts = false;
    int mMaxFolded = 0, mMaxOverlapping = 0;
    Vec3 mSphereCenter = Vec3::Zero();
    float mSphereRadius = 0.0f;
    bool mHasSphere = false;

    void advectMesh(double t, float dt);
    // The level set rebuilt from the mesh (semi-Lagrangian), then the iso-surface.
    void rebuildLevelSet();
};
}  // namespace springls
