#include <springls/Ply.hpp>

#include <happly.h>

#include <array>
#include <exception>
#include <fstream>
#include <vector>

namespace springls
{
namespace
{
void AddPoints(happly::Element &element, const Points3 &points, const std::array<const char *, 3> &names)
{
    std::vector<float> c(points.size());
    for (int axis = 0; axis < 3; axis++) {
        for (size_t i = 0; i < points.size(); i++) c[i] = points[i][axis];
        element.addProperty<float>(names[axis], c);
    }
}

bool HasPoints(happly::Element &element, const std::array<const char *, 3> &names)
{
    return element.hasProperty(names[0]) && element.hasProperty(names[1]) && element.hasProperty(names[2]);
}

Points3 GetPoints(happly::Element &element, const std::array<const char *, 3> &names)
{
    const std::vector<float> x = element.getProperty<float>(names[0]), y = element.getProperty<float>(names[1]),
                             z = element.getProperty<float>(names[2]);
    Points3 points(x.size());
    for (size_t i = 0; i < points.size(); i++) points[i] = Vec3(x[i], y[i], z[i]);
    return points;
}

constexpr std::array<const char *, 3> kPosition{"x", "y", "z"};
constexpr std::array<const char *, 3> kNormal{"nx", "ny", "nz"};
constexpr std::array<const char *, 3> kParticle{"px", "py", "pz"};

bool Write(const std::filesystem::path &file, happly::PLYData &ply, bool binary, std::string &error)
{
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        error = "could not write " + file.string();
        return false;
    }
    ply.write(out, binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
    if (!out) {
        error = "could not write " + file.string();
        return false;
    }
    return true;
}

bool Open(const std::filesystem::path &file, std::ifstream &in, std::string &error)
{
    in.open(file, std::ios::binary);
    if (!in) error = "could not open " + file.string();
    return static_cast<bool>(in);
}
}  // namespace

bool WriteMeshPly(const std::filesystem::path &file, const SurfaceMesh3D &mesh, bool binary, std::string &error)
{
    try {
        happly::PLYData ply;
        ply.addElement("vertex", mesh.vertexes.size());
        happly::Element &vertexes = ply.getElement("vertex");
        AddPoints(vertexes, mesh.vertexes, kPosition);
        if (mesh.normals.size() == mesh.vertexes.size()) AddPoints(vertexes, mesh.normals, kNormal);
        const size_t K = static_cast<size_t>(mesh.elementSize());
        std::vector<std::vector<uint32_t>> faces(mesh.faceCount());
        for (size_t f = 0; f < faces.size(); f++) faces[f].assign(mesh.face(f), mesh.face(f) + K);
        ply.addElement("face", faces.size());
        ply.getElement("face").addListProperty<uint32_t>("vertex_indices", faces);
        return Write(file, ply, binary, error);
    } catch (const std::exception &e) {
        error = file.string() + ": " + e.what();
        return false;
    }
}

bool ReadMeshPly(const std::filesystem::path &file, SurfaceMesh3D &mesh, std::string &error)
{
    try {
        std::ifstream in;
        if (!Open(file, in, error)) return false;
        happly::PLYData ply(in);
        happly::Element &vertexes = ply.getElement("vertex");
        SurfaceMesh3D result;
        result.vertexes = GetPoints(vertexes, kPosition);
        const std::vector<std::vector<uint32_t>> faces = ply.hasElement("face") ? ply.getFaceIndices<uint32_t>() : std::vector<std::vector<uint32_t>>{};
        result.type = !faces.empty() && faces[0].size() == 3 ? ElementType::Triangle : ElementType::Quad;
        const size_t K = static_cast<size_t>(result.elementSize());
        result.indexes.reserve(faces.size() * K);
        for (const std::vector<uint32_t> &face : faces) {
            if (face.size() != K) {
                error = file.string() + ": faces are not all triangles or all quads";
                return false;
            }
            for (uint32_t v : face) {
                if (v >= result.vertexes.size()) {
                    error = file.string() + ": face vertex index out of range";
                    return false;
                }
            }
            result.indexes.insert(result.indexes.end(), face.begin(), face.end());
        }
        if (HasPoints(vertexes, kNormal)) {
            result.normals = GetPoints(vertexes, kNormal);
        } else {
            result.updateVertexNormals();
        }
        mesh = std::move(result);
        return true;
    } catch (const std::exception &e) {
        error = file.string() + ": " + e.what();
        return false;
    }
}

bool WriteSpringlsPly(const std::filesystem::path &file, const Manifold3D &manifold, bool binary, std::string &error)
{
    try {
        happly::PLYData ply;
        ply.addElement("vertex", manifold.springlVertexes.size());
        AddPoints(ply.getElement("vertex"), manifold.springlVertexes, kPosition);
        const size_t count = manifold.springlCount();
        std::vector<std::vector<uint32_t>> faces(count);
        for (size_t n = 0; n < count; n++) {
            faces[n].resize(static_cast<size_t>(manifold.springlSize(n)));
            for (size_t i = 0; i < faces[n].size(); i++) faces[n][i] = manifold.springlBegin(n) + static_cast<uint32_t>(i);
        }
        ply.addElement("face", count);
        happly::Element &face = ply.getElement("face");
        face.addListProperty<uint32_t>("vertex_indices", faces);
        AddPoints(face, manifold.springlParticles, kParticle);
        if (manifold.springlParticleNormals.size() == count) AddPoints(face, manifold.springlParticleNormals, kNormal);
        return Write(file, ply, binary, error);
    } catch (const std::exception &e) {
        error = file.string() + ": " + e.what();
        return false;
    }
}

bool ReadSpringlsPly(const std::filesystem::path &file, Manifold3D &manifold, std::string &error)
{
    try {
        std::ifstream in;
        if (!Open(file, in, error)) return false;
        happly::PLYData ply(in);
        const Points3 points = GetPoints(ply.getElement("vertex"), kPosition);
        const std::vector<std::vector<uint32_t>> faces = ply.hasElement("face") ? ply.getFaceIndices<uint32_t>() : std::vector<std::vector<uint32_t>>{};
        Manifold3D result;
        Vec3 springl[kMaxSpringlSize];
        for (const std::vector<uint32_t> &face : faces) {
            if (face.size() < 3 || face.size() > static_cast<size_t>(kMaxSpringlSize)) {
                error = file.string() + ": springl with " + std::to_string(face.size()) + " vertexes (3 to " + std::to_string(kMaxSpringlSize) + " allowed)";
                return false;
            }
            Vec3 centroid = Vec3::Zero();
            for (size_t i = 0; i < face.size(); i++) {
                if (face[i] >= points.size()) {
                    error = file.string() + ": face vertex index out of range";
                    return false;
                }
                springl[i] = points[face[i]];
                centroid += springl[i];
            }
            result.springlParticles.push_back(centroid / static_cast<float>(face.size()));
            result.appendSpringlVertexes(std::span<const Vec3>(springl, face.size()));
        }
        if (ply.hasElement("face")) {
            happly::Element &face = ply.getElement("face");
            if (HasPoints(face, kParticle)) result.springlParticles = GetPoints(face, kParticle);
            if (HasPoints(face, kNormal)) result.springlParticleNormals = GetPoints(face, kNormal);
        }
        if (result.springlParticleNormals.size() != result.springlCount()) result.updateNormals();
        manifold.springlParticles = std::move(result.springlParticles);
        manifold.springlVertexes = std::move(result.springlVertexes);
        manifold.springlStart = std::move(result.springlStart);
        manifold.vertexSpringl = std::move(result.vertexSpringl);
        manifold.springlParticleNormals = std::move(result.springlParticleNormals);
        return true;
    } catch (const std::exception &e) {
        error = file.string() + ": " + e.what();
        return false;
    }
}
}  // namespace springls
