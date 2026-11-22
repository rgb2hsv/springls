#include <mesh/MeshDistance3D.hpp>

#include <boxtree.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace springls
{
namespace
{
uint64_t EdgeKey(uint32_t a, uint32_t b)
{
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | b;
}
}  // namespace

struct MeshDistance3D::Locator {
    libkdtree::BoxTree<libkdtree::Triangle<3>> tree;
};

MeshDistance3D::MeshDistance3D() = default;
MeshDistance3D::~MeshDistance3D() = default;
MeshDistance3D::MeshDistance3D(MeshDistance3D &&) noexcept = default;
MeshDistance3D &MeshDistance3D::operator=(MeshDistance3D &&) noexcept = default;

bool MeshDistance3D::build(const Points3 &positions, const std::vector<std::array<uint32_t, 3>> &triangles)
{
    mFaces.clear();
    mLocator.reset();
    mVertexNormals.assign(positions.size(), Vec3::Zero());
    std::unordered_map<uint64_t, Vec3> edgeSums;
    edgeSums.reserve(triangles.size() * 2);
    mFaces.reserve(triangles.size());
    std::vector<libkdtree::Triangle<3>> boxTriangles;
    boxTriangles.reserve(triangles.size());

    for (const auto &t : triangles) {
        if (t[0] >= positions.size() || t[1] >= positions.size() || t[2] >= positions.size()) {
            // Corrupt input would leave holes and plausible but wrong signs.
            mFaces.clear();
            return false;
        }
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
        const Vec3 corners[3] = {positions[t[0]], positions[t[1]], positions[t[2]]};
        const Vec3 cross = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
        const float area2 = cross.norm();
        const float longest =
            std::max({(corners[1] - corners[0]).squaredNorm(), (corners[2] - corners[1]).squaredNorm(), (corners[0] - corners[2]).squaredNorm()});
        // Zero-area (and non-finite) triangles carry no usable normal.
        if (!std::isfinite(area2) || !(area2 > 1e-12f * longest)) continue;

        Face face;
        face.v = t;
        face.p = {corners[0], corners[1], corners[2]};
        face.lo = corners[0].cwiseMin(corners[1]).cwiseMin(corners[2]);
        face.hi = corners[0].cwiseMax(corners[1]).cwiseMax(corners[2]);
        face.normal = cross / area2;
        for (int i = 0; i < 3; i++) {
            const Vec3 e1 = corners[(i + 1) % 3] - corners[i];
            const Vec3 e2 = corners[(i + 2) % 3] - corners[i];
            const float angle = std::atan2(e1.cross(e2).norm(), e1.dot(e2));
            mVertexNormals[t[i]] += angle * face.normal;
            edgeSums.try_emplace(EdgeKey(t[i], t[(i + 1) % 3]), Vec3::Zero()).first->second += face.normal;
        }
        const auto toBox = [](const Vec3 &v) { return libkdtree::Vec<3>{v.x(), v.y(), v.z()}; };
        boxTriangles.push_back({toBox(corners[0]), toBox(corners[1]), toBox(corners[2])});
        mFaces.push_back(face);
    }
    if (mFaces.empty()) return false;
    for (Face &face : mFaces) {
        for (int i = 0; i < 3; i++) face.edgeNormal[i] = edgeSums[EdgeKey(face.v[i], face.v[(i + 1) % 3])];
    }
    // BoxTree reorders its primitives but index() maps a hit back to the position in boxTriangles, i.e. the face id.
    mLocator = std::make_unique<Locator>();
    mLocator->tree.build(std::move(boxTriangles));
    return true;
}

float MeshDistance3D::signedDistance(const Vec3 &p, float maxDistance) const
{
    // The sign needs the nearest triangle even beyond maxDistance, so the search is unbounded.
    if (!mLocator) return maxDistance;
    const auto hit = mLocator->tree.nearest(libkdtree::Vec<3>{p.x(), p.y(), p.z()});
    if (!hit) return maxDistance;
    // The tree only picks the face; the float distance, closest point and region (which selects the pseudonormal)
    // come from the same computation as the candidate-list query.
    const Face &face = mFaces[mLocator->tree.index(hit->primitive)];
    Vec3 q;
    int region;
    const float d2 = TriangleDistanceSqr(face.p[0], face.p[1], face.p[2], p, &q, &region);
    return signFromFeature(face, region, p, q, std::min(std::sqrt(d2), maxDistance));
}

std::optional<float> MeshDistance3D::signedDistance(const Vec3 &p, std::span<const uint32_t> faces, float maxDistance, uint32_t *closestFace) const
{
    // The same search as TriangleTree3D::nearest() over a list instead of the tree.
    float best = std::isinf(maxDistance) ? std::numeric_limits<float>::infinity() : maxDistance * maxDistance;
    int bestFace = -1, bestRegion = kRegionFace;
    Vec3 bestPoint = p;
    for (const uint32_t f : faces) {
        if (f >= mFaces.size()) continue;
        const Face &face = mFaces[f];
        // The box bounds the distance from below; equality still needs the exact test for the tie rule.
        if ((p - p.cwiseMax(face.lo).cwiseMin(face.hi)).squaredNorm() > best) continue;
        Vec3 q;
        int region;
        const float d2 = TriangleDistanceSqr(face.p[0], face.p[1], face.p[2], p, &q, &region);
        if (d2 < best || (d2 == best && bestFace >= 0 && static_cast<int>(f) < bestFace)) {
            best = d2;
            bestFace = static_cast<int>(f);
            bestPoint = q;
            bestRegion = region;
        }
    }
    if (bestFace < 0) return std::nullopt;
    if (closestFace) *closestFace = static_cast<uint32_t>(bestFace);
    return signFromFeature(mFaces[bestFace], bestRegion, p, bestPoint, std::sqrt(best));
}

bool MeshDistance3D::faceNearBox(size_t face, const Vec3 &lo, const Vec3 &hi, float radius) const
{
    const Face &f = mFaces[face];
    // Separated along an axis, or along the face normal (both tests only reject what is farther than radius;
    // the slack keeps float rounding from rejecting a face right at it).
    const float r = radius + 1e-5f * (1.0f + radius + hi.cwiseAbs().maxCoeff() + lo.cwiseAbs().maxCoeff());
    if (((f.lo.array() - r) > hi.array()).any() || ((f.hi.array() + r) < lo.array()).any()) return false;
    const Vec3 center = 0.5f * (lo + hi), half = 0.5f * (hi - lo);
    const float reach = f.normal.cwiseAbs().dot(half);
    return std::abs(f.normal.dot(center - f.p[0])) <= reach + r;
}

float MeshDistance3D::signFromFeature(const Face &face, int region, const Vec3 &p, const Vec3 &closest, float distance) const
{
    // The feature holding the closest point comes straight from the distance computation; recomputing it
    // from the rounded closest point would misclassify edge hits on small triangles far from the origin.
    Vec3 normal = face.normal;
    switch (region) {
        case kRegionVertexA: normal = mVertexNormals[face.v[0]]; break;
        case kRegionVertexB: normal = mVertexNormals[face.v[1]]; break;
        case kRegionVertexC: normal = mVertexNormals[face.v[2]]; break;
        case kRegionEdgeAB: normal = face.edgeNormal[0]; break;
        case kRegionEdgeBC: normal = face.edgeNormal[1]; break;
        case kRegionEdgeCA: normal = face.edgeNormal[2]; break;
        default: break;
    }
    // Opposing faces on a non-manifold edge can cancel; the face normal is the fallback.
    if (normal.squaredNorm() < 1e-10f) normal = face.normal;
    return (p - closest).dot(normal) < 0.0f ? -distance : distance;
}
}  // namespace springls
