#include <mesh/Icosphere.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace springls
{
LoopMesh MakeIcosphere(int levels, double radius, const Vec3d &center)
{
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<Vec3d> v = {{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                            {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    for (Vec3d &p : v) p.normalize();
    std::vector<std::array<uint32_t, 3>> f = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                              {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
                                              {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};
    for (int l = 0; l < levels; l++) {
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
        next.reserve(4 * f.size());
        for (const auto &tri : f) {
            const uint32_t a = midpoint(tri[0], tri[1]), b = midpoint(tri[1], tri[2]), c = midpoint(tri[2], tri[0]);
            next.push_back({tri[0], a, c});
            next.push_back({tri[1], b, a});
            next.push_back({tri[2], c, b});
            next.push_back({a, b, c});
        }
        f = std::move(next);
    }
    LoopMesh mesh;
    for (const Vec3d &p : v) mesh.vertices.push_back((center + radius * p).cast<float>());
    mesh.triangles = std::move(f);
    return mesh;
}
}  // namespace springls
