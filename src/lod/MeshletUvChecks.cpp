#include <lod/MeshletUvChecks.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>

namespace springls
{
namespace
{
using Triangle2 = std::array<Vec2d, 3>;

Triangle2 UvTriangle(const MeshletLevel &level, size_t f) { return {level.texcoords[3 * f], level.texcoords[3 * f + 1], level.texcoords[3 * f + 2]}; }

double DoubleArea(const Vec2d &a, const Vec2d &b, const Vec2d &c) { return (b - a).x() * (c - a).y() - (b - a).y() * (c - a).x(); }

std::array<double, 3> Angles(const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Vector3d &c)
{
    const auto angle = [](const Eigen::Vector3d &u, const Eigen::Vector3d &v) { return std::atan2(u.cross(v).norm(), u.dot(v)); };
    return {angle(b - a, c - a), angle(c - b, a - b), angle(a - c, b - c)};
}

// Pairs of triangles whose interiors overlap, found through a uniform grid of their bounding boxes.
void CountOverlaps(const MeshletLevel &level, MeshletUvChecks &checks)
{
    const size_t F = level.triangles.size();
    if (F < 2) return;
    Vec2d lo = Vec2d::Constant(std::numeric_limits<double>::infinity()), hi = -lo;
    double meanSize = 0.0;
    for (size_t f = 0; f < F; f++) {
        const Triangle2 t = UvTriangle(level, f);
        for (const Vec2d &v : t) {
            lo = lo.cwiseMin(v);
            hi = hi.cwiseMax(v);
        }
        meanSize += std::sqrt(std::abs(0.5 * DoubleArea(t[0], t[1], t[2])));
    }
    meanSize /= static_cast<double>(F);
    const double cell = std::max(2.0 * meanSize, 1e-12);
    const auto cellOf = [&](double v, double origin) { return static_cast<long long>(std::floor((v - origin) / cell)); };
    std::unordered_map<long long, std::vector<uint32_t>> grid;
    const long long columns = cellOf(hi.x(), lo.x()) + 1;
    for (size_t f = 0; f < F; f++) {
        const Triangle2 t = UvTriangle(level, f);
        Vec2d a = t[0].cwiseMin(t[1]).cwiseMin(t[2]), b = t[0].cwiseMax(t[1]).cwiseMax(t[2]);
        for (long long y = cellOf(a.y(), lo.y()); y <= cellOf(b.y(), lo.y()); y++) {
            for (long long x = cellOf(a.x(), lo.x()); x <= cellOf(b.x(), lo.x()); x++) grid[y * columns + x].push_back(static_cast<uint32_t>(f));
        }
    }
    std::map<std::pair<uint32_t, uint32_t>, bool> tested;
    for (const auto &[key, faces] : grid) {
        for (size_t i = 0; i < faces.size(); i++) {
            for (size_t j = i + 1; j < faces.size(); j++) {
                const std::pair<uint32_t, uint32_t> pair = std::minmax(faces[i], faces[j]);
                if (!tested.emplace(pair, true).second) continue;
                const Triangle2 p = UvTriangle(level, pair.first), q = UvTriangle(level, pair.second);
                const double texels = UvOverlapDepth(p, q) * kOverlapCheckTextureSize;
                checks.maxOverlapTexels = std::max(checks.maxOverlapTexels, texels);
                if (texels <= 1.0) continue;
                checks.overlaps++;
                checks.crossRegionOverlaps += level.faceRegion[pair.first] != level.faceRegion[pair.second];
                checks.overlappingTriangles.push_back(pair.first);
                checks.overlappingTriangles.push_back(pair.second);
            }
        }
    }
    std::sort(checks.overlappingTriangles.begin(), checks.overlappingTriangles.end());
    checks.overlappingTriangles.erase(std::unique(checks.overlappingTriangles.begin(), checks.overlappingTriangles.end()),
                                      checks.overlappingTriangles.end());
}
}  // namespace

std::vector<MeshletCornerPair> MeshletCornerPairs(const MeshletLevel &level)
{
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edges;  // edge -> faces
    for (uint32_t f = 0; f < level.triangles.size(); f++) {
        const auto &t = level.triangles[f];
        for (int c = 0; c < 3; c++) edges[std::minmax(t[c], t[(c + 1) % 3])].push_back(f);
    }
    std::vector<MeshletCornerPair> pairs;
    const auto cornerOf = [&](uint32_t f, uint32_t v) {
        const auto &t = level.triangles[f];
        return t[0] == v ? 0 : t[1] == v ? 1 : 2;
    };
    for (const auto &[edge, faces] : edges) {
        if (faces.size() != 2 || level.faceRegion[faces[0]] != level.faceRegion[faces[1]]) continue;
        for (uint32_t v : {edge.first, edge.second}) pairs.push_back({faces[0], faces[1], cornerOf(faces[0], v), cornerOf(faces[1], v)});
    }
    return pairs;
}

MeshletUvChecks CheckMeshletUvs(const MeshletLevel &level)
{
    MeshletUvChecks checks;
    const size_t F = level.triangles.size();
    checks.triangles = F;
    if (level.texcoords.size() != 3 * F) return checks;
    std::map<int, std::vector<double>> regionRatios;
    std::vector<double> ratio(F, 0.0);
    for (size_t f = 0; f < F; f++) {
        const auto &t = level.triangles[f];
        const Triangle2 uv = UvTriangle(level, f);
        const Eigen::Vector3d a = level.vertexes[t[0]].cast<double>(), b = level.vertexes[t[1]].cast<double>(), c = level.vertexes[t[2]].cast<double>();
        const double uvArea = 0.5 * DoubleArea(uv[0], uv[1], uv[2]);
        if (uvArea <= 0.0) checks.flips++;
        const auto uvAngles = Angles({uv[0].x(), uv[0].y(), 0.0}, {uv[1].x(), uv[1].y(), 0.0}, {uv[2].x(), uv[2].y(), 0.0});
        const auto xyzAngles = Angles(a, b, c);
        for (int k = 0; k < 3; k++) checks.maxAngleError = std::max(checks.maxAngleError, std::abs(uvAngles[k] - xyzAngles[k]));
        const double xyzArea = 0.5 * (b - a).cross(c - a).norm();
        ratio[f] = xyzArea > 0.0 ? std::abs(uvArea) / xyzArea : 0.0;
        regionRatios[level.faceRegion[f]].push_back(ratio[f]);
    }
    std::map<int, double> median;
    for (auto &[region, values] : regionRatios) {
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        median[region] = values[values.size() / 2];
    }
    checks.minAreaRatio = std::numeric_limits<double>::infinity();
    checks.maxAreaRatio = 0.0;
    for (size_t f = 0; f < F; f++) {
        const double m = median[level.faceRegion[f]];
        const double r = m > 0.0 ? ratio[f] / m : 0.0;
        checks.minAreaRatio = std::min(checks.minAreaRatio, r);
        checks.maxAreaRatio = std::max(checks.maxAreaRatio, r);
    }
    const std::vector<MeshletCornerPair> pairs = MeshletCornerPairs(level);
    double sum = 0.0;
    for (const MeshletCornerPair &p : pairs) {
        const double gap = (level.texcoords[3 * p.faceA + p.cornerA] - level.texcoords[3 * p.faceB + p.cornerB]).norm();
        sum += gap;
        checks.maxCornerGap = std::max(checks.maxCornerGap, gap);
    }
    checks.meanCornerGap = pairs.empty() ? 0.0 : sum / pairs.size();
    CountOverlaps(level, checks);
    return checks;
}
}  // namespace springls
