#pragma once
#include <springls/Volume.hpp>

#include <vector>

namespace springls
{
// Polygon mesh with one element type: face f has vertexes indexes[K f .. K f + K - 1], K =
// ElementSize(type), wound counterclockwise about the outward normal (toward phi > 0).
struct SurfaceMesh3D {
    ElementType type = ElementType::Quad;
    Points3 vertexes;
    Points3 normals;
    std::vector<uint32_t> indexes;

    int elementSize() const { return ElementSize(type); }
    size_t faceCount() const { return indexes.size() / static_cast<size_t>(elementSize()); }
    const uint32_t *face(size_t f) const { return indexes.data() + f * static_cast<size_t>(elementSize()); }
    void clear()
    {
        vertexes.clear();
        normals.clear();
        indexes.clear();
    }
    // Area-weighted vertex normals from the faces.
    void updateVertexNormals();
    // cereal serialization (checkpoints).
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(type, vertexes, normals, indexes);
    }
};

// Zero iso-surface of phi, searched over the voxels in `active` (the narrow band).
//
// Quads come from surface nets [Gibson, "Constrained Elastic Surface Nets", MICCAI 1998]: one
// vertex per grid cell the surface passes through, at the mean of the cell's edge crossings, and
// one quad per crossed grid edge joining the four cells around it. Triangles split each quad
// along its shorter diagonal, so both element types describe the same surface. Edges whose cells
// would leave the grid are skipped. Output order depends only on phi, not on `active`'s order.
void ExtractIsoSurface(const VolumeF &phi, const std::vector<Vec3i> &active, ElementType type, SurfaceMesh3D &mesh);
}  // namespace springls
