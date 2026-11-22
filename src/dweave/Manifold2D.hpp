#pragma once
#include <core/Common.hpp>

namespace springls
{
// Springl constellation of a DoubleWeave springl.
//   Primal: particles centered on iso-contour edge midpoints.
//   Dual:   particles centered on iso-contour vertexes, between neighboring primal particles.
enum class Constellation : uint8_t {
    Primal = 0,
    Dual = 1
};
constexpr size_t kConstellationCount = 2;

// Surface state of a spring level set: the iso-contour and the springls that carry it.
// Springl n has particle springlParticles[n] and arm vertexes springlVertexes[2n], [2n+1].
struct Manifold2D {
    // Iso-contour of the level set.
    Points2 vertexLocations;
    Curves vertexIndexes;
    // Springls.
    Points2 springlParticles;
    Points2 springlVertexes;
    Points2 springlParticleNormals;
    // Constellation of each springl (DoubleWeave only; empty otherwise).
    std::vector<uint8_t> springlConstellations;
    // Region label of each springl (multi-object methods only; empty otherwise).
    std::vector<int> springlLabels;
    // Region label of each curve of vertexIndexes (multi-object methods only; empty otherwise).
    std::vector<int> curveLabels;
    // Index of the springl each one descends from in the previous step (tracking).
    std::vector<int> springlParticleTracking;
    // Latest particle velocity (handed to new springls by tracking).
    Points2 velocities;
    // Position of each springl's particle on the initial interface (tracking).
    Points2 correspondence;

    size_t springlCount() const { return springlParticles.size(); }
    // Normal of springl n: the arm direction v1 - v0 rotated +90 degrees, which points outward
    // because the contour keeps the inside on its right.
    void updateNormals();
};
}  // namespace springls
