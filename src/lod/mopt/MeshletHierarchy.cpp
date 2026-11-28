// The only translation unit of springls that includes meshlet-optimizer (see CMakeLists.txt, springls_lod).
#include <flattening/IsometricUvs.hpp>
#include <lod/MeshletHierarchy.hpp>

#include <hierarchy/MeshletBuilder.hpp>

#include <exception>

namespace springls
{
std::vector<MeshletLevel> BuildMeshletHierarchy(const LoopMesh &mesh, const MeshletHierarchyParams &params, const std::function<Vec3(const Vec3 &)> &project)
{
    mopt::Mesh input;
    input.vertexLocations.reserve(mesh.vertices.size());
    for (const Vec3 &p : mesh.vertices) input.vertexLocations.emplace_back(p.x(), p.y(), p.z());
    input.triIndexes.reserve(mesh.triangles.size());
    for (const auto &t : mesh.triangles) input.triIndexes.emplace_back(t[0], t[1], t[2]);
    input.updateVertexNormals();

    mopt::MeshletParams settings;
    settings.maxDepth = params.maxDepth;
    settings.randomSeed = params.randomSeed;
    settings.isometricUvs = true;
    settings.uvRelaxSteps = params.uvRelaxSteps;
    settings.uvFitToAtlas = params.uvFitToAtlas;
    settings.uvEnforceNoOverlaps = params.uvEnforceNoOverlaps;
    mopt::MeshletBuilder builder(settings);
    if (project) {
        builder.projectToSurface = [&project](const mopt::Vector3f &p) {
            const Vec3 q = project(Vec3(p.x(), p.y(), p.z()));
            return mopt::Vector3f(q.x(), q.y(), q.z());
        };
    }
    try {
        builder.build(input);
    } catch (const std::exception &) {
        return {};
    }

    std::vector<MeshletLevel> levels(builder.levels.size());
    for (size_t k = 0; k < levels.size(); k++) {
        const mopt::LodLevel &from = builder.levels[k];
        MeshletLevel &to = levels[k];
        to.vertexes.reserve(from.simplified.vertexLocations.size());
        for (const mopt::Vector3f &p : from.simplified.vertexLocations) to.vertexes.emplace_back(p.x(), p.y(), p.z());
        to.triangles.reserve(from.simplified.triIndexes.size());
        for (const mopt::Vector3u &t : from.simplified.triIndexes) to.triangles.push_back({t.x(), t.y(), t.z()});
        to.faceRegion = from.simplifiedFaceRegion;
        to.vertexRegion = from.simplifiedVertexRegion;
        to.texcoords.assign(from.isometricUvs.begin(), from.isometricUvs.end());
        to.atlasTexcoords.assign(from.atlasUvs.begin(), from.atlasUvs.end());
        if (k < builder.regionHierarchy.parent.size() && k + 1 < levels.size()) to.regionParent = builder.regionHierarchy.parent[k];
    }
    return levels;
}
double UvOverlapDepth(const std::array<Vec2d, 3> &p, const std::array<Vec2d, 3> &q)
{
    return mopt::OverlapOf({p[0], p[1], p[2]}, {q[0], q[1], q[2]}).depth;
}

}  // namespace springls
