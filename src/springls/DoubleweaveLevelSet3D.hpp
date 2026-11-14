#pragma once
#include <springls/SpringLevelSet3D.hpp>

#include <array>
#include <functional>

namespace springls
{
// Calls dual(particle, vertexes, v) for every interior vertex v of `mesh`: the dual springl is v's
// dual cell, the polygon through the centroids of the faces around v (in order, wound like the
// faces), with its particle at v. It sits where the corners of the primal springls meet, its
// corners reach toward the primal particles, and it has one corner per face around v (four on a
// regular quad mesh; any valence up to kMaxSpringlSize). The dual springls tile the surface exactly
// once, like the faces, staggered from them by half a face. Boundary and non-manifold vertexes are
// skipped.
void ForEachDualSpringl(const SurfaceMesh3D &mesh, const std::function<void(const Vec3 &particle, std::span<const Vec3> vertexes, uint32_t vertex)> &dual);

// Doubleweave springls in 3D: two interleaved springl constellations on one level set (3D
// counterpart of DoubleweaveLevelSet2D).
//
// The primal constellation seeds one springl per iso-surface face (particle at the face centroid),
// exactly like SpringLevelSet3D; the dual constellation seeds the dual springls of the iso-surface
// (see ForEachDualSpringl). Attraction, gap filling, contraction and tracking act within a
// constellation. Both constellations are advected by the same flow and together define the
// unsigned distance field of the combined level set (the output surface). Each constellation also
// keeps its own level set and iso-surface, evolved from its own springls alone, and fill /
// contract test each constellation against its own iso-surface, so a hole in one constellation is
// detected even where the other still holds the combined surface in place.
class DoubleweaveLevelSet3D : public SpringLevelSet3D
{
public:
    explicit DoubleweaveLevelSet3D(ElementType elementType = ElementType::Quad);
    bool init() override;
    size_t countSpringls(Constellation constellation) const;

protected:
    // One constellation's level set and iso-surface, evolved from that constellation's springls only.
    struct ConstellationLevelSet {
        NarrowBand3D band;
        VolumeF unsignedLevelSet;
        SurfaceMesh3D isoSurface;
        bool valid = false;
    };
    std::array<ConstellationLevelSet, kConstellationCount> mConstellations;
    // Constellation swapped into the base-class members, or -1 for the combined one.
    int mActiveConstellation = -1;
    std::vector<uint8_t> mOldConstellations;
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;

    // Exchanges the base-class level set / iso-surface with constellation w's.
    void swapConstellation(size_t w);
    void ensureConstellationLevelSets();
    void evolveConstellationLevelSets();
    size_t appendSpringl(const Vec3 &particle, std::span<const Vec3> vertexes, Constellation constellation);

    void seedSpringls() override;
    void clearSpringls() override;
    void compactSpringls(const std::vector<uint8_t> &keep) override;
    void snapshotForTracking() override;
    bool canTrackFrom(size_t newIndex, size_t oldIndex) const override;
    bool canAttract(size_t springl, size_t other) const override { return mManifold.springlConstellations[springl] == mManifold.springlConstellations[other]; }
    bool refinesContourWith(size_t particle) const override;
    int fill() override;
    int contract() override;
    void afterLevelSetUpdate() override { evolveConstellationLevelSets(); }
};
}  // namespace springls
