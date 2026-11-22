#pragma once
#include <core/Common3D.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace springls
{
// Exact signed distance to a closed triangle mesh (negative inside). The sign comes from angle-weighted
// pseudonormals [Baerentzen & Aanaes 2005]; edge pseudonormals sum all incident faces, so non-manifold
// edges are handled. Queries are const and allocation free, so they are safe to run concurrently.
class MeshDistance3D
{
public:
    MeshDistance3D();
    ~MeshDistance3D();
    MeshDistance3D(MeshDistance3D &&) noexcept;
    MeshDistance3D &operator=(MeshDistance3D &&) noexcept;

    // Returns false when there are no non-degenerate triangles or any index is out of range (the object is
    // then empty). Zero-area and repeated-index triangles are skipped.
    bool build(const Points3 &positions, const std::vector<std::array<uint32_t, 3>> &triangles);
    // Signed distance (negative inside), |d| clamped to maxDistance, sign exact everywhere.
    // Returns maxDistance (positive) when the object is empty, i.e. build() was not called or failed.
    float signedDistance(const Vec3 &p, float maxDistance = std::numeric_limits<float>::infinity()) const;

    // Faces are the non-degenerate input triangles build() kept, in input order (0 .. faceCount() - 1).
    size_t faceCount() const { return mFaces.size(); }
    void faceBounds(size_t face, Vec3 &lo, Vec3 &hi) const
    {
        lo = mFaces[face].lo;
        hi = mFaces[face].hi;
    }
    // Conservative: true whenever the face comes within `radius` of the box (may be true slightly beyond).
    bool faceNearBox(size_t face, const Vec3 &lo, const Vec3 &hi, float radius) const;
    // Closest of `faces` within maxDistance (strictly): its signed distance, with the sign from the same
    // pseudonormals as the full query, or nullopt if none is that close. Matches signedDistance(p) whenever
    // `faces` holds the closest face (ties go to the smaller face, as in the tree).
    // closestFace, if given, receives the face it found.
    std::optional<float> signedDistance(const Vec3 &p, std::span<const uint32_t> faces, float maxDistance, uint32_t *closestFace = nullptr) const;
    // Unsigned distance from p to one face (the same computation the queries use).
    float faceDistance(size_t face, const Vec3 &p) const
    {
        const Face &f = mFaces[face];
        return std::sqrt(TriangleDistanceSqr(f.p[0], f.p[1], f.p[2], p));
    }

private:
    struct Face {
        std::array<uint32_t, 3> v;
        std::array<Vec3, 3> p;           // corner positions
        Vec3 lo, hi;                     // bounds
        Vec3 normal;                     // unit face normal
        std::array<Vec3, 3> edgeNormal;  // pseudonormal of edge (v[i], v[i + 1])
    };
    std::vector<Face> mFaces;
    Points3 mVertexNormals;  // angle-weighted pseudonormal per input vertex
    // kdtree-cpp BoxTree over the faces; it only locates the nearest face (hidden so the header stays light).
    struct Locator;
    std::unique_ptr<Locator> mLocator;

    // Signed distance of p, whose closest point `closest` lies on feature `region` (TriangleRegion) of `face`.
    float signFromFeature(const Face &face, int region, const Vec3 &p, const Vec3 &closest, float distance) const;
};
}  // namespace springls
