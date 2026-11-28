// MeshCheck3D: fold-overs and overlaps of a triangle mesh.
#include "Check.hpp"

#include <lod/MeshCheck3D.hpp>
#include <mesh/Icosphere.hpp>

#include <cstdio>

using namespace springls;

namespace
{
TriangleSurface Sphere(const Vec3 &c, float r, int levels)
{
    const LoopMesh m = MakeIcosphere(levels, r, c.cast<double>());
    TriangleSurface mesh;
    for (const auto &p : m.vertices) mesh.vertexes.push_back(p);
    mesh.triangles = m.triangles;
    mesh.faceRegion.assign(mesh.triangles.size(), 0);
    return mesh;
}
}  // namespace

TEST(triangles_intersect)
{
    const Vec3 a0(0, 0, 0), a1(1, 0, 0), a2(0, 1, 0);
    CHECK(TrianglesIntersect(a0, a1, a2, Vec3(0.2f, 0.2f, -1), Vec3(0.2f, 0.2f, 1), Vec3(0.3f, 0.3f, 1)));   // pierces
    CHECK(!TrianglesIntersect(a0, a1, a2, Vec3(0.2f, 0.2f, 0.1f), Vec3(1, 0.2f, 0.1f), Vec3(0.2f, 1, 0.1f)));  // above
    CHECK(!TrianglesIntersect(a0, a1, a2, Vec3(0.6f, 0.6f, -1), Vec3(0.6f, 0.6f, 1), Vec3(1, 1, 0)));        // beside
    CHECK(TrianglesIntersect(a0, a1, a2, Vec3(0.1f, 0.1f, 0), Vec3(0.5f, 0.1f, 0), Vec3(0.1f, 0.5f, 0)));     // coplanar, inside
    CHECK(!TrianglesIntersect(a0, a1, a2, Vec3(1, 1, 0), Vec3(2, 1, 0), Vec3(1, 2, 0)));                     // coplanar, apart
}

TEST(mesh_check_sphere_is_clean)
{
    const MeshCheck check = CheckMesh(Sphere(Vec3(0, 0, 0), 2.0f, 3));
    std::printf("  folded %d, overlapping %d, non-manifold %d\n", check.foldedEdges, check.overlappingPairs, check.nonManifoldEdges);
    CHECK(check.clean());
}

// A vertex slid along the surface past one of its neighbours: the triangles between them flip over.
TEST(mesh_check_finds_fold)
{
    TriangleSurface mesh = Sphere(Vec3(0, 0, 0), 2.0f, 3);
    const uint32_t v = mesh.triangles[0][0], n = mesh.triangles[0][1];
    const Vec3 past = mesh.vertexes[n] + 0.6f * (mesh.vertexes[n] - mesh.vertexes[v]);
    mesh.vertexes[v] = 2.0f * past.normalized();
    const MeshCheck check = CheckMesh(mesh);
    std::printf("  folded %d, overlapping %d\n", check.foldedEdges, check.overlappingPairs);
    CHECK(check.foldedEdges > 0);
    int folded = 0;
    for (uint8_t f : check.flags) folded += (f & MeshCheck::Folded) != 0;
    CHECK(folded > 0 && folded <= 20);
}

// A vertex pushed right through the sphere: its triangles pass through the far side.
TEST(mesh_check_finds_overlap)
{
    TriangleSurface mesh = Sphere(Vec3(0, 0, 0), 2.0f, 3);
    mesh.vertexes[0] = -1.3f * mesh.vertexes[0];
    const MeshCheck check = CheckMesh(mesh);
    std::printf("  folded %d, overlapping %d\n", check.foldedEdges, check.overlappingPairs);
    CHECK(check.overlappingPairs > 0);
}

// Two spheres passing through each other overlap along a circle and nowhere else.
TEST(mesh_check_finds_intersecting_spheres)
{
    TriangleSurface a = Sphere(Vec3(0, 0, 0), 2.0f, 3);
    const TriangleSurface b = Sphere(Vec3(2.5f, 0, 0), 2.0f, 3);
    const uint32_t offset = static_cast<uint32_t>(a.vertexes.size());
    a.vertexes.insert(a.vertexes.end(), b.vertexes.begin(), b.vertexes.end());
    for (auto t : b.triangles) a.triangles.push_back({t[0] + offset, t[1] + offset, t[2] + offset});
    const MeshCheck check = CheckMesh(a);
    std::printf("  folded %d, overlapping %d of %zu triangles\n", check.foldedEdges, check.overlappingPairs, a.triangles.size());
    CHECK(check.foldedEdges == 0 && check.overlappingPairs > 0);
    for (size_t f = 0; f < a.triangles.size(); f++) {
        if (!check.flags[f]) continue;
        const Vec3 &p = a.vertexes[a.triangles[f][0]];
        CHECK(std::abs(p.x() - 1.25f) < 0.6f);  // near the plane of the circle where they meet
    }
}
