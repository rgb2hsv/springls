#include "Check.hpp"

#include <io/Ply.hpp>
#include <mesh/MeshDistance3D.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

using namespace springls;

namespace
{
const float kCube[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
const int kCubeQuads[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {3, 0, 4, 7}};

// Appends an outward-wound cube with minimum corner `offset`, reusing vertexes already at the same position.
void AddCube(PolygonMesh &mesh, const Vec3 &offset)
{
    uint32_t ids[8];
    for (int i = 0; i < 8; i++) {
        const Vec3 p = offset + Vec3(kCube[i][0], kCube[i][1], kCube[i][2]);
        ids[i] = static_cast<uint32_t>(mesh.positions.size());
        for (size_t j = 0; j < mesh.positions.size(); j++) {
            if (mesh.positions[j] == p) ids[i] = static_cast<uint32_t>(j);
        }
        if (ids[i] == mesh.positions.size()) mesh.positions.push_back(p);
    }
    for (const auto &quad : kCubeQuads) {
        for (int k : quad) mesh.faceVertexes.push_back(ids[k]);
        mesh.faceStart.push_back(static_cast<uint32_t>(mesh.faceVertexes.size()));
    }
}

const float kTestPi = std::numbers::pi_v<float>;

// Returns the id of the vertex at p, appending it when absent (exact match).
uint32_t VertexAt(PolygonMesh &mesh, const Vec3 &p)
{
    for (size_t j = 0; j < mesh.positions.size(); j++) {
        if (mesh.positions[j] == p) return static_cast<uint32_t>(j);
    }
    mesh.positions.push_back(p);
    return static_cast<uint32_t>(mesh.positions.size() - 1);
}

void AddFace(PolygonMesh &mesh, const std::vector<Vec3> &corners)
{
    for (const Vec3 &c : corners) mesh.faceVertexes.push_back(VertexAt(mesh, c));
    mesh.faceStart.push_back(static_cast<uint32_t>(mesh.faceVertexes.size()));
}

// Closed prism over a convex counterclockwise polygon extruded from z = 0 to z = 1 (outward wound).
void AddPrism(PolygonMesh &mesh, const std::vector<std::array<float, 2>> &polygon)
{
    std::vector<Vec3> top, bottom;
    for (const auto &q : polygon) {
        bottom.push_back(Vec3(q[0], q[1], 0.0f));
        top.push_back(Vec3(q[0], q[1], 1.0f));
    }
    std::reverse(bottom.begin(), bottom.end());
    AddFace(mesh, bottom);
    AddFace(mesh, top);
    const size_t n = polygon.size();
    for (size_t i = 0; i < n; i++) {
        const size_t j = (i + 1) % n;
        AddFace(mesh, {Vec3(polygon[i][0], polygon[i][1], 0.0f), Vec3(polygon[j][0], polygon[j][1], 0.0f), Vec3(polygon[j][0], polygon[j][1], 1.0f),
                       Vec3(polygon[i][0], polygon[i][1], 1.0f)});
    }
}

Vec3 Place(const Vec3 &v, float scale, float offset)
{
    const Eigen::AngleAxisf rotation(0.7f, Vec3(1.0f, 2.0f, 3.0f).normalized());
    return rotation * (v * scale) + Vec3(offset, 0.6f * offset, -0.3f * offset);
}

// Generalized winding number of the triangle mesh at p (about 1 inside, 0 outside, double precision).
double WindingNumber(const Points3 &positions, const std::vector<std::array<uint32_t, 3>> &triangles, const Vec3 &p)
{
    double total = 0.0;
    for (const auto &t : triangles) {
        const Eigen::Vector3d a = (positions[t[0]] - p).cast<double>(), b = (positions[t[1]] - p).cast<double>(), c = (positions[t[2]] - p).cast<double>();
        const double la = a.norm(), lb = b.norm(), lc = c.norm();
        const double numerator = a.dot(b.cross(c));
        const double denominator = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
        total += 2.0 * std::atan2(numerator, denominator);
    }
    return total / (4.0 * std::numbers::pi);
}

void CheckCubeDistances(const MeshDistance3D &distance)
{
    CHECK_NEAR(distance.signedDistance(Vec3(0.5f, 0.5f, 0.5f)), -0.5, 1e-5);
    CHECK_NEAR(distance.signedDistance(Vec3(2.0f, 0.5f, 0.5f)), 1.0, 1e-5);
    CHECK_NEAR(distance.signedDistance(Vec3(2.0f, 2.0f, 2.0f)), std::sqrt(3.0), 1e-5);
    CHECK_NEAR(distance.signedDistance(Vec3(1.5f, 1.5f, 0.5f)), std::sqrt(0.5), 1e-5);
}
}  // namespace

TEST(distance_cube_inside_outside)
{
    PolygonMesh mesh;
    AddCube(mesh, Vec3::Zero());
    MeshDistance3D distance;
    CHECK(distance.build(mesh.positions, Triangulate(mesh)));
    CheckCubeDistances(distance);
}

TEST(distance_clamps_magnitude)
{
    PolygonMesh mesh;
    AddCube(mesh, Vec3::Zero());
    MeshDistance3D distance;
    CHECK(distance.build(mesh.positions, Triangulate(mesh)));
    CHECK_NEAR(distance.signedDistance(Vec3(5.0f, 0.5f, 0.5f), 1.0f), 1.0, 1e-6);
    CHECK_NEAR(distance.signedDistance(Vec3(0.5f, 0.5f, 0.5f), 0.1f), -0.1, 1e-6);
}

TEST(distance_nonmanifold_bowtie)
{
    PolygonMesh mesh;
    AddCube(mesh, Vec3::Zero());
    AddCube(mesh, Vec3(1.0f, 1.0f, 0.0f));
    CHECK(mesh.positions.size() == 14);
    CHECK(AnalyzeTopology(mesh).nonManifoldEdges == 1);
    MeshDistance3D distance;
    CHECK(distance.build(mesh.positions, Triangulate(mesh)));
    CHECK(distance.signedDistance(Vec3(0.5f, 0.5f, 0.5f)) < 0.0f);
    CHECK(distance.signedDistance(Vec3(1.5f, 1.5f, 0.5f)) < 0.0f);
    CHECK(distance.signedDistance(Vec3(1.5f, 0.5f, 0.5f)) > 0.0f);
    CHECK(distance.signedDistance(Vec3(0.5f, 1.5f, 0.5f)) > 0.0f);
    CHECK(distance.signedDistance(Vec3(1.02f, 1.02f, 0.5f)) < 0.0f);
    CHECK(distance.signedDistance(Vec3(1.02f, 0.98f, 0.5f)) > 0.0f);
}

TEST(distance_degenerate_faces)
{
    PolygonMesh mesh;
    AddCube(mesh, Vec3::Zero());
    auto triangles = Triangulate(mesh);
    mesh.positions.push_back(Vec3(9.0f, 9.0f, 9.0f));  // unreferenced vertex
    triangles.push_back({0, 1, 1});                    // repeated index
    const uint32_t first = static_cast<uint32_t>(mesh.positions.size());
    mesh.positions.push_back(Vec3(0.25f, 0.25f, 0.0f));
    mesh.positions.push_back(Vec3(0.5f, 0.25f, 0.0f));
    mesh.positions.push_back(Vec3(0.75f, 0.25f, 0.0f));
    triangles.push_back({first, first + 1, first + 2});  // collinear, zero area
    MeshDistance3D distance;
    CHECK(distance.build(mesh.positions, triangles));
    CheckCubeDistances(distance);
}

TEST(distance_only_degenerate_fails)
{
    MeshDistance3D distance;
    CHECK(!distance.build(Points3{Vec3::Zero(), Vec3::Ones()}, {{0, 1, 1}}));
}

// A thin wedge rotated off axis and placed far from the origin: points around its sharp apex edge are
// closest to that edge, and the edge pseudonormal must decide the sign despite rounding in the hit point.
TEST(distance_sharp_crease_offset)
{
    const float s = std::tan(5.5f * kTestPi / 180.0f);
    for (const float scale : {1.0f, 0.1f, 0.01f}) {
        for (const float offset : {0.0f, 10.0f, 100.0f}) {
            PolygonMesh mesh;
            AddPrism(mesh, {{0.0f, 0.0f}, {1.0f, -s}, {1.0f, s}});
            for (Vec3 &v : mesh.positions) v = Place(v, scale, offset);
            MeshDistance3D distance;
            CHECK(distance.build(mesh.positions, Triangulate(mesh)));
            int wrong = 0;
            for (int i = 0; i <= 60; i++) {
                const float phi = (-80.0f + 160.0f * i / 60.0f) * kTestPi / 180.0f;
                for (const float r : {0.1f, 0.3f, 1.0f}) {
                    for (const float z : {0.2f, 0.5f, 0.8f}) {
                        const Vec3 outside = Place(Vec3(-r * std::cos(phi), r * std::sin(phi), z), scale, offset);
                        if (!(distance.signedDistance(outside) > 0.0f)) wrong++;
                    }
                }
            }
            CHECK(wrong == 0);
            CHECK(distance.signedDistance(Place(Vec3(0.5f, 0.0f, 0.5f), scale, offset)) < 0.0f);
        }
    }
}

// Two wedges share their apex edge (4 faces on it) with a gap of more than 180 degrees between the face
// planes; points in that gap are closest to the shared edge or its end vertexes.
TEST(distance_nonmanifold_edge_region)
{
    const auto at = [](float degrees) {
        const float a = degrees * kTestPi / 180.0f;
        return std::array<float, 2>{std::cos(a), std::sin(a)};
    };
    PolygonMesh mesh;
    AddPrism(mesh, {{0.0f, 0.0f}, at(-10.0f), at(10.0f)});
    AddPrism(mesh, {{0.0f, 0.0f}, at(40.0f), at(60.0f)});
    CHECK(AnalyzeTopology(mesh).nonManifoldEdges == 1);
    MeshDistance3D distance;
    CHECK(distance.build(mesh.positions, Triangulate(mesh)));
    for (int deg = 90; deg <= 330; deg += 10) {
        const float a = static_cast<float>(deg) * kTestPi / 180.0f;
        for (const float r : {0.05f, 0.5f}) CHECK(distance.signedDistance(Vec3(r * std::cos(a), r * std::sin(a), 0.5f)) > 0.0f);
    }
    // Closest feature is the shared vertex (0, 0, 0) or (0, 0, 1).
    CHECK(distance.signedDistance(Vec3(-0.3f, -0.2f, -0.4f)) > 0.0f);
    CHECK(distance.signedDistance(Vec3(-0.3f, -0.2f, 1.4f)) > 0.0f);
    CHECK_NEAR(distance.signedDistance(Vec3(-0.3f, -0.2f, -0.4f)), std::sqrt(0.09 + 0.04 + 0.16), 1e-5);
    // Inside each wedge, and in the small gap between them.
    CHECK(distance.signedDistance(Vec3(0.5f, 0.0f, 0.5f)) < 0.0f);
    CHECK(distance.signedDistance(Vec3(0.5f * std::cos(50.0f * kTestPi / 180.0f), 0.5f * std::sin(50.0f * kTestPi / 180.0f), 0.5f)) < 0.0f);
    CHECK(distance.signedDistance(Vec3(0.3f * std::cos(25.0f * kTestPi / 180.0f), 0.3f * std::sin(25.0f * kTestPi / 180.0f), 0.5f)) > 0.0f);
}

// Random points around a non-convex (L shaped), rotated and offset solid against a winding number.
TEST(distance_matches_winding_number_on_nonconvex_mesh)
{
    // The L polygon is star shaped about its first corner, so the caps are fanned from it.
    const std::vector<std::array<float, 2>> polygon = {{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}};
    Points3 positions;
    std::vector<std::array<uint32_t, 3>> triangles;
    for (const auto &q : polygon) positions.push_back(Place(Vec3(q[0], q[1], 0.0f), 0.5f, 10.0f));
    for (const auto &q : polygon) positions.push_back(Place(Vec3(q[0], q[1], 1.0f), 0.5f, 10.0f));
    const uint32_t n = static_cast<uint32_t>(polygon.size());
    for (uint32_t i = 1; i + 1 < n; i++) {
        triangles.push_back({0, i + 1, i});          // bottom, facing -z
        triangles.push_back({n, n + i, n + i + 1});  // top, facing +z
    }
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t j = (i + 1) % n;
        triangles.push_back({i, j, n + j});
        triangles.push_back({i, n + j, n + i});
    }
    MeshDistance3D distance;
    CHECK(distance.build(positions, triangles));
    CHECK(WindingNumber(positions, triangles, Place(Vec3(0.5f, 0.5f, 0.5f), 0.5f, 10.0f)) > 0.99);

    uint32_t state = 12345u;
    const auto random = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / 16777216.0f;
    };
    int inside = 0, outside = 0, mismatches = 0, skipped = 0;
    for (int i = 0; i < 20000; i++) {
        const Vec3 local(-0.5f + 3.0f * random(), -0.5f + 3.0f * random(), -0.5f + 2.0f * random());
        const Vec3 p = Place(local, 0.5f, 10.0f);
        const float d = distance.signedDistance(p);
        if (std::abs(d) < 1e-3f) {
            skipped++;
            continue;
        }
        const bool truth = WindingNumber(positions, triangles, p) > 0.5;
        if ((d < 0.0f) != truth) mismatches++;
        (truth ? inside : outside)++;
    }
    CHECK(mismatches == 0);
    CHECK(inside > 500);
    CHECK(outside > 500);
    CHECK(skipped < 100);
}

TEST(distance_rejects_out_of_range_index)
{
    PolygonMesh mesh;
    AddCube(mesh, Vec3::Zero());
    auto triangles = Triangulate(mesh);
    triangles.push_back({0, 1, 99});
    MeshDistance3D distance;
    CHECK(!distance.build(mesh.positions, triangles));
    CHECK(distance.signedDistance(Vec3(0.5f, 0.5f, 0.5f), 2.0f) == 2.0f);
}
