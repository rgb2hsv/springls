#pragma once
#include <dweave/Manifold2D.hpp>
#include <levelset/IsoSurface3D.hpp>

#include <span>

namespace springls
{
// Surface state of a 3D spring level set: the iso-surface and the springls that carry it.
// Springl n has particle springlParticles[n] and arm vertexes
// springlVertexes[springlStart[n] .. springlStart[n + 1]), at least three and at most
// kMaxSpringlSize, wound counterclockwise about its outward normal. Primal springls have
// elementSize() vertexes; dual (DoubleWeave) springls have one per face around their vertex.
struct Manifold3D {
    // Iso-surface of the level set (its element type is the springls' element type).
    SurfaceMesh3D isoSurface;
    // Springls.
    ElementType elementType = ElementType::Quad;
    Points3 springlParticles;
    Points3 springlVertexes;
    std::vector<uint32_t> springlStart{0};  // springlCount() + 1 offsets into springlVertexes
    std::vector<uint32_t> vertexSpringl;    // springl of each entry of springlVertexes
    Points3 springlParticleNormals;
    // Constellation of each springl (DoubleWeave only; empty otherwise).
    std::vector<uint8_t> springlConstellations;
    // Index of the springl each one descends from at the start of the latest step (tracking).
    std::vector<int> springlParticleTracking;
    // Latest particle velocity (handed to new springls by tracking).
    Points3 velocities;
    // Position of each springl's particle on the initial interface (tracking).
    Points3 correspondence;

    // Vertexes per iso-surface face and per primal springl.
    int elementSize() const { return ElementSize(elementType); }
    size_t springlCount() const { return springlParticles.size(); }
    uint32_t springlBegin(size_t n) const { return springlStart[n]; }
    int springlSize(size_t n) const { return static_cast<int>(springlStart[n + 1] - springlStart[n]); }
    const Vec3 *springl(size_t n) const { return springlVertexes.data() + springlStart[n]; }
    // Removes every springl (particles, vertexes and offsets; per-springl attributes are the caller's).
    void clearSpringlVertexes()
    {
        springlVertexes.clear();
        springlStart.assign(1, 0);
        vertexSpringl.clear();
    }
    // Appends the vertexes of springl springlCount() - 1 (call after pushing its particle).
    void appendSpringlVertexes(std::span<const Vec3> vertexes)
    {
        const uint32_t n = static_cast<uint32_t>(springlStart.size() - 1);
        springlVertexes.insert(springlVertexes.end(), vertexes.begin(), vertexes.end());
        vertexSpringl.insert(vertexSpringl.end(), vertexes.size(), n);
        springlStart.push_back(static_cast<uint32_t>(springlVertexes.size()));
    }
    // Normal of springl n: the normalized fan normal of its element about its particle.
    void updateNormals();
    // cereal serialization (checkpoints).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(isoSurface, elementType, springlParticles, springlVertexes, springlStart, vertexSpringl, springlParticleNormals, springlConstellations,
           springlParticleTracking, velocities, correspondence);
    }
};
}  // namespace springls
