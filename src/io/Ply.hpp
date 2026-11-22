#pragma once
#include <core/Common3D.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace springls
{
// Polygon soup read from a file: positions plus faces of any valence stored as offsets.
struct PolygonMesh {
    Points3 positions;
    std::vector<uint32_t> faceStart{0};  // faceCount()+1 offsets into faceVertexes
    std::vector<uint32_t> faceVertexes;
    size_t faceCount() const { return faceStart.size() - 1; }
};

// Reads an ASCII or binary (either byte order) PLY file with happly. Returns false and sets error on failure.
bool LoadPly(const std::filesystem::path &file, PolygonMesh &mesh, std::string &error);

// Splits each polygon into a triangle fan around its first vertex.
std::vector<std::array<uint32_t, 3>> Triangulate(const PolygonMesh &mesh);

// Mean length over the unique edges of the polygons (0 when there are none).
float MeanEdgeLength(const PolygonMesh &mesh);

struct MeshTopology {
    size_t boundaryEdges = 0;     // edges used by exactly one polygon
    size_t nonManifoldEdges = 0;  // edges used by more than two polygons
};

MeshTopology AnalyzeTopology(const PolygonMesh &mesh);

// A PLY file as tables, for arbitrary elements and properties: header comments, and per element each
// property's values as doubles (exact for every PLY type but 64-bit integers, which PLY has none of).
enum class PlyType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64
};
struct PlyProperty {
    std::string name;
    PlyType type = PlyType::Float32;
    bool isList = false;
    std::vector<double> values;          // scalars: one per row; lists: every item, row after row
    std::vector<uint32_t> listStart{0};  // lists only: rows + 1 offsets into values
    size_t listSize(size_t row) const { return listStart[row + 1] - listStart[row]; }
};
struct PlyElement {
    std::string name;
    size_t count = 0;
    std::vector<PlyProperty> properties;
    const PlyProperty *find(const std::string &property) const;
};
struct PlyTable {
    std::vector<std::string> comments;  // without the "comment " keyword
    std::vector<PlyElement> elements;
    const PlyElement *find(const std::string &element) const;
};
// Writes binary little-endian PLY with happly (without its signature comment). Every property of an element
// must have count values (or count lists); list lengths are written as uchar, so a list holds at most 255 items.
bool WritePly(std::ostream &out, const PlyTable &table, std::string &error);
// Reads ASCII or binary (either byte order) PLY with happly.
bool ReadPly(std::istream &in, PlyTable &table, std::string &error);
}  // namespace springls
