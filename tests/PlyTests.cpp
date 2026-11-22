#include "Check.hpp"

#include <io/Ply.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

using namespace springls;

namespace
{
const float kCube[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
const int kCubeQuads[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {3, 0, 4, 7}};

std::filesystem::path WriteFile(const std::string &name, const std::string &content)
{
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    return path;
}

// ASCII cube; line ending, extra header lines, coordinate type, extra vertex properties (and their
// values) and the face list types are variable. skipFace removes one quad to open the mesh.
std::string CubeAscii(const std::string &eol, const std::string &extraHeader, const std::string &coordType, const std::string &vertexExtraProps,
                      const std::string &vertexSuffix, const std::string &listTypes, int skipFace = -1)
{
    std::ostringstream s;
    const int faces = skipFace >= 0 ? 5 : 6;
    s << "ply" << eol << "format ascii 1.0" << eol << extraHeader;
    s << "element vertex 8" << eol << "property " << coordType << " x" << eol << "property " << coordType << " y" << eol << "property " << coordType << " z"
      << eol << vertexExtraProps;
    s << "element face " << faces << eol << "property list " << listTypes << eol << "end_header" << eol;
    for (const auto &v : kCube) s << v[0] << " " << v[1] << " " << v[2] << vertexSuffix << eol;
    for (int f = 0; f < 6; ++f) {
        if (f == skipFace) continue;
        s << "4 " << kCubeQuads[f][0] << " " << kCubeQuads[f][1] << " " << kCubeQuads[f][2] << " " << kCubeQuads[f][3] << eol;
    }
    return s.str();
}

void CheckCube(const PolygonMesh &mesh)
{
    CHECK(mesh.positions.size() == 8);
    CHECK(mesh.faceCount() == 6);
    CHECK(Triangulate(mesh).size() == 12);
    CHECK_NEAR(MeanEdgeLength(mesh), 1.0f, 1e-6);
    const MeshTopology topology = AnalyzeTopology(mesh);
    CHECK(topology.boundaryEdges == 0);
    CHECK(topology.nonManifoldEdges == 0);
}
}  // namespace

TEST(ply_ascii_cube_quads)
{
    const auto path = WriteFile("springls_cube_ascii.ply", CubeAscii("\n", "", "float", "", "", "uchar int vertex_indices"));
    PolygonMesh mesh;
    std::string error;
    CHECK(LoadPly(path, mesh, error));
    CheckCube(mesh);
}

TEST(ply_binary_le_triangles)
{
    const float v[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const int32_t f[4][3] = {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}};
    std::string data =
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
        "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face 4\n"
        "property list uint8 int32 vertex_indices\nend_header\n";
    for (int i = 0; i < 4; ++i) {
        const float record[6] = {v[i][0], v[i][1], v[i][2], 0.f, 0.f, 1.f};
        data.append(reinterpret_cast<const char *>(record), sizeof(record));
    }
    for (int i = 0; i < 4; ++i) {
        data.push_back(3);
        data.append(reinterpret_cast<const char *>(f[i]), sizeof(f[i]));
    }
    const auto path = WriteFile("springls_tet_binary.ply", data);
    PolygonMesh mesh;
    std::string error;
    CHECK(LoadPly(path, mesh, error));
    CHECK(mesh.positions.size() == 4);
    CHECK(mesh.faceCount() == 4);
    if (mesh.positions.size() == 4 && mesh.faceCount() == 4) {
        for (int i = 0; i < 4; ++i) {
            CHECK(mesh.positions[i].x() == v[i][0]);
            CHECK(mesh.positions[i].y() == v[i][1]);
            CHECK(mesh.positions[i].z() == v[i][2]);
        }
        CHECK(mesh.faceVertexes[3] == 0 && mesh.faceVertexes[4] == 1 && mesh.faceVertexes[5] == 3);
    }
    CHECK(AnalyzeTopology(mesh).boundaryEdges == 0);
}

TEST(ply_header_variants)
{
    const std::string extra = "comment made by a test\r\nobj_info something\r\n";
    const std::string props = "property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n";
    const auto path = WriteFile("springls_cube_variants.ply", CubeAscii("\r\n", extra, "double", props, " 255 128 0", "uchar uint vertex_index"));
    PolygonMesh mesh;
    std::string error;
    CHECK(LoadPly(path, mesh, error));
    CheckCube(mesh);
}

TEST(ply_binary_big_endian)
{
    std::string data =
        "ply\nformat binary_big_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n";
    const auto putBigEndian = [&](auto value) {
        char bytes[sizeof(value)];
        std::memcpy(bytes, &value, sizeof(value));
        for (size_t i = sizeof(value); i-- > 0;) data.push_back(bytes[i]);
    };
    const float v[3][3] = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}};
    for (const auto &p : v)
        for (float c : p) putBigEndian(c);
    data.push_back(3);
    for (int32_t i : {0, 1, 2}) putBigEndian(i);
    const auto path = WriteFile("springls_big_endian.ply", data);
    PolygonMesh mesh;
    std::string error;
    CHECK(LoadPly(path, mesh, error));
    CHECK(mesh.positions.size() == 3 && mesh.faceCount() == 1);
    if (mesh.positions.size() == 3) CHECK(mesh.positions[1].x() == 2.0f && mesh.positions[2].y() == 3.0f);
    if (mesh.faceCount() == 1) CHECK(mesh.faceVertexes[1] == 1 && mesh.faceVertexes[2] == 2);
}

TEST(ply_rejects_malformed_and_missing)
{
    PolygonMesh mesh;
    const std::string header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    const std::string truncated = header + std::string(20, '\0');  // 36 bytes are needed
    const std::string cases[] = {
        "not a ply file\n",
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n",                                                         // no end_header
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",  // a row short
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0\n",    // a value short
        truncated,
        CubeAscii("\n", "", "float", "", "", "uchar int vertex_indices").replace(0, 3, "plx"),
    };
    for (size_t i = 0; i < std::size(cases); i++) {
        const auto path = WriteFile("springls_malformed_" + std::to_string(i) + ".ply", cases[i]);
        std::string error;
        CHECK(!LoadPly(path, mesh, error));
        CHECK(!error.empty());
    }

    std::string error;
    CHECK(!LoadPly(std::filesystem::temp_directory_path() / "springls_does_not_exist.ply", mesh, error));
    CHECK(!error.empty());
}

TEST(ply_table_round_trip)
{
    PlyTable table;
    PlyElement &vertex = table.elements.emplace_back();
    vertex.name = "vertex";
    vertex.count = 2;
    for (const char *axis : {"x", "y", "z"}) {
        PlyProperty &p = vertex.properties.emplace_back();
        p.name = axis;
        p.values = {0.5, -1.25};
    }
    PlyElement &face = table.elements.emplace_back();
    face.name = "face";
    face.count = 2;
    face.properties.resize(2);
    PlyProperty &indices = face.properties[0];
    indices.name = "vertex_indices";
    indices.type = PlyType::Int32;
    indices.isList = true;
    indices.values = {0, 1, 0, 1, 1, 0, 1};
    indices.listStart = {0, 3, 7};
    PlyProperty &label = face.properties[1];
    label.name = "label";
    label.type = PlyType::UInt8;
    label.values = {7, 255};

    std::stringstream bytes;
    std::string error;
    CHECK(WritePly(bytes, table, error));
    CHECK(bytes.str().find("comment") == std::string::npos);  // no happly signature
    PlyTable back;
    CHECK(ReadPly(bytes, back, error));
    CHECK(back.comments.empty());
    const PlyElement *f = back.find("face");
    CHECK(f && f->count == 2);
    if (f) {
        const PlyProperty *i = f->find("vertex_indices"), *l = f->find("label");
        CHECK(i && i->isList && i->type == PlyType::Int32 && i->values == indices.values && i->listStart == indices.listStart);
        CHECK(l && !l->isList && l->type == PlyType::UInt8 && l->values == label.values);
    }
    const PlyElement *v = back.find("vertex");
    CHECK(v && v->find("y") && v->find("y")->values == std::vector<double>({0.5, -1.25}));

    // Lists are written with uchar lengths.
    indices.values.assign(259, 0.0);
    indices.listStart = {0, 3, 259};
    CHECK(!WritePly(bytes, table, error));
}

TEST(ply_open_mesh_topology)
{
    const auto path = WriteFile("springls_cube_open.ply", CubeAscii("\n", "", "float", "", "", "uchar int vertex_indices", 1));
    PolygonMesh mesh;
    std::string error;
    CHECK(LoadPly(path, mesh, error));
    CHECK(mesh.faceCount() == 5);
    const MeshTopology topology = AnalyzeTopology(mesh);
    CHECK(topology.boundaryEdges == 4);
    CHECK(topology.nonManifoldEdges == 0);
}
