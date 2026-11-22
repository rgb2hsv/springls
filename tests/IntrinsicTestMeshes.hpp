#pragma once
// Closed test meshes for the intrinsic-triangulation tests: icospheres, a skewed icosphere with
// slivers, and a flat two-sided square grid.
#include <core/Common3D.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace springls_test
{
struct TestMesh {
    springls::Points3 vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
};

// Icosahedron subdivided `level` times, projected onto the sphere of `radius` (counter-clockwise
// faces seen from outside).
inline TestMesh Icosphere(int level, double radius = 1.0)
{
    using springls::Vec3d;
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Vec3d> v = {{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                            {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    for (Vec3d &p : v) p.normalize();
    std::vector<std::array<uint32_t, 3>> f = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                              {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
                                              {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};
    for (int l = 0; l < level; l++) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
        const auto midpoint = [&](uint32_t a, uint32_t b) {
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            const auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            v.push_back((v[a] + v[b]).normalized());
            const uint32_t id = static_cast<uint32_t>(v.size() - 1);
            mid[key] = id;
            return id;
        };
        std::vector<std::array<uint32_t, 3>> next;
        for (const auto &tri : f) {
            const uint32_t a = midpoint(tri[0], tri[1]), b = midpoint(tri[1], tri[2]), c = midpoint(tri[2], tri[0]);
            next.push_back({tri[0], a, c});
            next.push_back({tri[1], b, a});
            next.push_back({tri[2], c, b});
            next.push_back({a, b, c});
        }
        f = std::move(next);
    }
    TestMesh mesh;
    for (const Vec3d &p : v) mesh.vertices.push_back((radius * p).cast<float>());
    mesh.triangles = std::move(f);
    return mesh;
}

// Icosphere stretched along x (many slivers and obtuse corners).
inline TestMesh SkewedIcosphere(int level, double stretch = 5.0)
{
    TestMesh mesh = Icosphere(level);
    for (auto &p : mesh.vertices) p.x() *= static_cast<float>(stretch);
    return mesh;
}

// A flat n x n square grid of spacing h in the z = 0 plane, glued to a second copy of itself
// (reversed orientation) along the border: a closed two-sided sheet. Every square is split by the
// same diagonal into two right triangles. Intrinsically flat everywhere but at the four corners.
// The top sheet's vertex (i, j) is index j * (n + 1) + i, at (i h, j h, 0), counter-clockwise seen from +z.
inline TestMesh DoubledGrid(int n, double h = 1.0)
{
    TestMesh mesh;
    const int N = n + 1;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) mesh.vertices.emplace_back(static_cast<float>(i * h), static_cast<float>(j * h), 0.0f);
    }
    const auto top = [N](int i, int j) { return static_cast<uint32_t>(j * N + i); };
    std::vector<uint32_t> bottomIndex(N * N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            const bool border = i == 0 || j == 0 || i == n || j == n;
            if (border) {
                bottomIndex[j * N + i] = top(i, j);
            } else {
                bottomIndex[j * N + i] = static_cast<uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(mesh.vertices[top(i, j)]);
            }
        }
    }
    const auto bottom = [&](int i, int j) { return bottomIndex[j * N + i]; };
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            // The two corner squares whose usual diagonal would join two border vertices (an edge of
            // both sheets) use the other diagonal.
            if ((i == n - 1 && j == 0) || (i == 0 && j == n - 1)) {
                mesh.triangles.push_back({top(i, j), top(i + 1, j), top(i, j + 1)});
                mesh.triangles.push_back({top(i + 1, j), top(i + 1, j + 1), top(i, j + 1)});
                mesh.triangles.push_back({bottom(i, j), bottom(i, j + 1), bottom(i + 1, j)});
                mesh.triangles.push_back({bottom(i + 1, j), bottom(i, j + 1), bottom(i + 1, j + 1)});
                continue;
            }
            mesh.triangles.push_back({top(i, j), top(i + 1, j), top(i + 1, j + 1)});
            mesh.triangles.push_back({top(i, j), top(i + 1, j + 1), top(i, j + 1)});
            mesh.triangles.push_back({bottom(i, j), bottom(i + 1, j + 1), bottom(i + 1, j)});
            mesh.triangles.push_back({bottom(i, j), bottom(i, j + 1), bottom(i + 1, j + 1)});
        }
    }
    return mesh;
}
}  // namespace springls_test
