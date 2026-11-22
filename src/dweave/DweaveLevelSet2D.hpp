#pragma once
#include <springl/SpringLevelSet2D.hpp>

#include <array>

namespace springls
{
// Iso-contour curve as an ordered vertex list; closed curves drop the repeated first vertex.
struct ContourCurve {
    std::vector<uint32_t> vertices;
    bool closed = false;
    explicit ContourCurve(const Curve &curve);
    // Calls edge(a, b) for every contour edge.
    template <class Fn>
    void forEachEdge(Fn &&edge) const
    {
        const size_t N = vertices.size();
        if (N < 2) return;
        const size_t edges = closed ? N : N - 1;
        for (size_t i = 0; i < edges; i++) edge(vertices[i], vertices[(i + 1) % N]);
    }
    // Calls corner(u, v, w) for every vertex v that has both neighbors u and w.
    template <class Fn>
    void forEachCorner(Fn &&corner) const
    {
        const size_t N = vertices.size();
        if (N < 3) return;
        if (closed) {
            for (size_t k = 0; k < N; k++) corner(vertices[(k + N - 1) % N], vertices[k], vertices[(k + 1) % N]);
        } else {
            for (size_t k = 1; k + 1 < N; k++) corner(vertices[k - 1], vertices[k], vertices[k + 1]);
        }
    }
};

// Builds the dual chord springl on contour vertex v with neighbors u and w: a segment parallel to
// the chord between the midpoints of (u, v) and (v, w), with the same length, translated so it
// passes through v. The particle is v unless v projects past a chord end (sharp corners), in which
// case it is that end. Returns false when the chord is degenerate.
bool MakeChordSpringl(const Vec2 &u, const Vec2 &v, const Vec2 &w, Vec2 &particle, Vec2 &v0, Vec2 &v1);

// DoubleWeave springls: two interleaved springl constellations on one level set.
//
// The primal constellation seeds one springl per iso-contour edge (particle at the edge midpoint,
// arms on the edge endpoints), exactly like SpringLevelSet2D. The dual constellation seeds one
// springl per iso-contour vertex (see MakeChordSpringl), so primal and dual particles are
// staggered by half an edge. Attraction, gap filling, contraction and tracking act within a
// constellation; both constellations are advected by the same flow and together define the
// unsigned distance field that drives the combined level set.
//
// Each constellation also keeps its own level set, evolved from its own springls alone. Fill and
// contract test each constellation against its own iso-contour, so a hole in one constellation is
// detected even where the other still holds the combined surface in place.
class DweaveLevelSet2D : public SpringLevelSet2D
{
public:
    struct DweaveTuning {
        // Fill each constellation first from the other's springls: joints between neighboring
        // springls of one constellation mark where the other's particles belong.
        bool fillFromOtherConstellation = false;
        // Contract every springl against the combined contour instead of its own constellation's.
        bool contractAgainstCombinedContour = false;
    };
    DweaveTuning dweaveTuning;

    DweaveLevelSet2D();
    bool init() override;
    size_t countSpringls(Constellation constellation) const;

protected:
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    // One constellation's level set and contour, evolved from that constellation's springls only.
    struct ConstellationLevelSet {
        LevelSet2D band;
        ImageF unsignedLevelSet;
        Points2 vertexLocations;
        Curves vertexIndexes;
        bool valid = false;
    };
    std::array<ConstellationLevelSet, kConstellationCount> mConstellations;
    // Constellation swapped into the base-class members, or -1 for the combined one.
    int mActiveConstellation = -1;
    std::vector<uint8_t> mOldConstellations;

    // Exchanges the base-class level set / contour with constellation w's.
    void swapConstellation(size_t w);
    void ensureConstellationLevelSets();
    // Unsigned distance to constellation w's springls (constellation w must be swapped in).
    void updateConstellationUnsignedLevelSet(size_t w);
    void evolveConstellationLevelSets();
    size_t appendSpringl(const Vec2 &particle, const Vec2 &v0, const Vec2 &v1, Constellation constellation);

    void seedSpringls() override;
    void clearSpringls() override;
    void compactSpringls(const std::vector<uint8_t> &keep) override;
    void snapshotForTracking() override;
    bool canTrackFrom(size_t newIndex, size_t oldIndex) const override;
    bool canAttract(size_t springl, size_t other) const override { return mManifold.springlConstellations[springl] == mManifold.springlConstellations[other]; }
    int fill() override;
    int contract() override;
    void afterLevelSetUpdate() override { evolveConstellationLevelSets(); }
    bool refinesContourWith(size_t particle) const override;
};
}  // namespace springls
