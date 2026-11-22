// Checkpoints of the 3D simulations (SpringLevelSet3D::saveState / loadState), written with cereal's
// portable binary archive. Every class writes its own members after its base's (writeState /
// readState); value types (Volume, NarrowBand3D, SurfaceMesh3D, Manifold3D, AdaptiveGrid3D, ...)
// carry cereal serialize / save / load member templates in their headers.
#include <dweave/DweaveLevelSet3D.hpp>
#include <io/Ply.hpp>
#include <io/Vdb.hpp>
#include <springl/AdaptiveSpringLevelSet3D.hpp>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <algorithm>
#include <exception>
#include <istream>
#include <limits>
#include <map>
#include <ostream>
#include <span>
#include <string>
#include <utility>

namespace Eigen
{
// Fixed-size Eigen vectors (Vec3, Vec3i) as their coefficients.
template <class Archive, class Scalar, int Rows>
void serialize(Archive &ar, Matrix<Scalar, Rows, 1> &v)
{
    for (int i = 0; i < Rows; i++) ar(v[i]);
}
}  // namespace Eigen

namespace springls
{
namespace
{
constexpr char kMagic[] = "springls-3d-state";
constexpr uint32_t kVersion = 3;  // 2: the semi-Lagrangian rebuild times are no longer stored; 3: level sets in a VDB file
const char *ElementTypeName(ElementType type) { return type == ElementType::Quad ? "quad" : "triangle"; }
// The VDB grid every checkpoint has (the combined unsigned level set), which carries the header.
const char *kHeaderGrid = "unsigned";
const char *kConstellationPrefix[kConstellationCount] = {"primal_", "dual_"};

// An adaptive grid, or nothing for an empty one (no "<prefix>blocks_d0" grid then).
bool WriteGrid(VdbWriter &writer, const AdaptiveGrid3D &grid, const std::string &prefix, std::string &error)
{
    return grid.leaves().empty() || grid.writeVdb(writer, prefix, error);
}

bool ReadGrid(const VdbReader &reader, AdaptiveGrid3D &grid, const std::string &prefix, std::string &error)
{
    if (!reader.find(prefix + "blocks_d0")) {
        grid.clear();
        return true;
    }
    return grid.readVdb(reader, prefix, error);
}
}  // namespace

bool SpringLevelSet3D::saveState(std::ostream &state, std::ostream &levelSets, std::string &error) const
{
    try {
        cereal::PortableBinaryOutputArchive ar(state);
        ar(std::string(kMagic), kVersion, mName, mManifold.elementType);
        writeState(ar);
    } catch (const std::exception &e) {
        error = std::string("could not write the state: ") + e.what();
        return false;
    }
    if (!state) {
        error = "could not write the state";
        return false;
    }
    // The VDB header ties the two files together: method, element type and the step they are of.
    VdbWriter writer;
    std::string vdbError;
    if (!writeLevelSets(writer, vdbError) || !writer.setMetadata(kHeaderGrid, "springls.magic", std::string(kMagic), vdbError) ||
        !writer.setMetadata(kHeaderGrid, "springls.version", static_cast<int32_t>(kVersion), vdbError) ||
        !writer.setMetadata(kHeaderGrid, "springls.method", mName, vdbError) ||
        !writer.setMetadata(kHeaderGrid, "springls.element", std::string(ElementTypeName(mManifold.elementType)), vdbError) ||
        !writer.setMetadata(kHeaderGrid, "springls.iteration", std::to_string(mIteration), vdbError) || !writer.write(levelSets, vdbError)) {
        error = "could not write the level sets: " + vdbError;
        return false;
    }
    return true;
}

bool SpringLevelSet3D::loadState(std::istream &state, std::istream &levelSets, std::string &error)
{
    try {
        cereal::PortableBinaryInputArchive ar(state);
        std::string magic, name;
        uint32_t version = 0;
        ElementType type = ElementType::Quad;
        ar(magic, version, name, type);
        if (magic != kMagic || version != kVersion) {
            error = "not a springls 3D state file (or a different version)";
            return false;
        }
        if (name != mName || type != mManifold.elementType) {
            error =
                "the state is of " + name + " with " + ElementTypeName(type) + " springls, not " + mName + " with " + ElementTypeName(mManifold.elementType);
            return false;
        }
        readState(ar);
    } catch (const std::exception &e) {
        error = std::string("could not read the state: ") + e.what();
        return false;
    }
    VdbReader reader;
    std::string vdbError;
    if (!reader.read(levelSets, vdbError)) {
        error = "could not read the level sets: " + vdbError;
        return false;
    }
    const VdbGrid *header = reader.find(kHeaderGrid);
    std::string magic, method, element, iteration;
    int32_t version = 0;
    if (!header || !header->metadata("springls.magic", magic) || !header->metadata("springls.version", version) ||
        !header->metadata("springls.method", method) || !header->metadata("springls.element", element) || !header->metadata("springls.iteration", iteration) ||
        magic != kMagic || version != static_cast<int32_t>(kVersion)) {
        error = "not a springls 3D level set file (or a different version)";
        return false;
    }
    if (method != mName || element != ElementTypeName(mManifold.elementType) || iteration != std::to_string(mIteration)) {
        error = "the level set file is of " + method + " with " + element + " springls at step " + iteration + ", not of this state (" + mName + " with " +
                ElementTypeName(mManifold.elementType) + " springls at step " + std::to_string(mIteration) + ")";
        return false;
    }
    if (!readLevelSets(reader, vdbError)) {
        error = "could not read the level sets: " + vdbError;
        return false;
    }
    return true;
}

void SpringLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    ar(mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight);
    const_cast<LevelSet3D &>(mBand).serializeBand(ar);
    ar(mManifold, mNearestNeighbors, mScaledSpringls, mSpringlScales, mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities,
       mOldVertexSpringl);
}

void SpringLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    ar(mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight);
    mBand.serializeBand(ar);
    ar(mManifold, mNearestNeighbors, mScaledSpringls, mSpringlScales, mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities,
       mOldVertexSpringl);
    mSpringlGrid.invalidate();
}

bool SpringLevelSet3D::writeLevelSets(VdbWriter &writer, std::string &error) const
{
    return mBand.writeVdb(writer, "", error) && WriteVolumeVdb(writer, kHeaderGrid, mUnsignedLevelSet, NarrowBand3D::kMaxDistance, false, error);
}

bool SpringLevelSet3D::readLevelSets(const VdbReader &reader, std::string &error)
{
    return mBand.readVdb(reader, "", error) && ReadVolumeVdb(reader, kHeaderGrid, mUnsignedLevelSet, error);
}

void DweaveLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet3D::writeState(ar);
    for (const ConstellationLevelSet &c : mConstellations) {
        const_cast<LevelSet3D &>(c.band).serializeBand(ar);
        ar(c.isoSurface, c.valid);
    }
    ar(mActiveConstellation, mOldConstellations);
}

void DweaveLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet3D::readState(ar);
    for (ConstellationLevelSet &c : mConstellations) {
        c.band.serializeBand(ar);
        ar(c.isoSurface, c.valid);
    }
    ar(mActiveConstellation, mOldConstellations);
}

bool DweaveLevelSet3D::writeLevelSets(VdbWriter &writer, std::string &error) const
{
    if (!SpringLevelSet3D::writeLevelSets(writer, error)) return false;
    for (size_t c = 0; c < kConstellationCount; c++) {
        const std::string prefix = kConstellationPrefix[c];
        if (!mConstellations[c].band.writeVdb(writer, prefix, error) ||
            !WriteVolumeVdb(writer, prefix + "unsigned", mConstellations[c].unsignedLevelSet, NarrowBand3D::kMaxDistance, false, error))
            return false;
    }
    return true;
}

bool DweaveLevelSet3D::readLevelSets(const VdbReader &reader, std::string &error)
{
    if (!SpringLevelSet3D::readLevelSets(reader, error)) return false;
    for (size_t c = 0; c < kConstellationCount; c++) {
        const std::string prefix = kConstellationPrefix[c];
        if (!mConstellations[c].band.readVdb(reader, prefix, error) || !ReadVolumeVdb(reader, prefix + "unsigned", mConstellations[c].unsignedLevelSet, error))
            return false;
    }
    return true;
}

void AdaptiveSpringLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet3D::writeState(ar);
    ar(mContourScales, mSplitOnRefine);
}

void AdaptiveSpringLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet3D::readState(ar);
    ar(mContourScales, mSplitOnRefine);
}

bool AdaptiveSpringLevelSet3D::writeLevelSets(VdbWriter &writer, std::string &error) const
{
    return SpringLevelSet3D::writeLevelSets(writer, error) && WriteGrid(writer, mGrid, "", error);
}

bool AdaptiveSpringLevelSet3D::readLevelSets(const VdbReader &reader, std::string &error)
{
    return SpringLevelSet3D::readLevelSets(reader, error) && ReadGrid(reader, mGrid, "", error);
}

void AdaptiveDweaveLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    AdaptiveSpringLevelSet3D::writeState(ar);
    for (const ConstellationState &c : mStates) ar(c.isoSurface, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

bool AdaptiveDweaveLevelSet3D::writeLevelSets(VdbWriter &writer, std::string &error) const
{
    if (!AdaptiveSpringLevelSet3D::writeLevelSets(writer, error)) return false;
    for (size_t c = 0; c < kConstellationCount; c++) {
        if (!WriteGrid(writer, mStates[c].grid, kConstellationPrefix[c], error)) return false;
    }
    return true;
}

bool AdaptiveDweaveLevelSet3D::readLevelSets(const VdbReader &reader, std::string &error)
{
    if (!AdaptiveSpringLevelSet3D::readLevelSets(reader, error)) return false;
    for (size_t c = 0; c < kConstellationCount; c++) {
        if (!ReadGrid(reader, mStates[c].grid, kConstellationPrefix[c], error)) return false;
    }
    return true;
}

void AdaptiveDweaveLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    AdaptiveSpringLevelSet3D::readState(ar);
    for (ConstellationState &c : mStates) ar(c.isoSurface, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

}  // namespace springls

// ---------------------------------------------------------------------------------------------
// The DoubleWeave state as separate parts (AdaptiveDweaveLevelSet3D::savePart / loadParts).
// ---------------------------------------------------------------------------------------------

namespace springls
{
namespace
{
constexpr char kPartMagic[] = "springls-3d-state-part";
constexpr uint32_t kPartVersion = 6;  // 2: level set .vdb, scalar state its own part; 3-5: springls .ply; 6: springls cereal again

// One constellation's springls in their global order, with every per-springl array that has one entry per
// springl (the others are left empty), and that constellation's own level set.
struct ConstellationPart {
    uint64_t total = 0;             // springls in the whole simulation
    std::vector<uint32_t> indexes;  // global index of each springl, ascending
    Points3 particles, normals, velocities, correspondence;
    std::vector<int> tracking;
    std::vector<float> scales;
    std::vector<uint32_t> sizes;  // arm vertexes per springl
    Points3 vertexes;
    std::vector<std::vector<SpringlEdge>> neighbors;  // per arm vertex, or empty
    bool hasVelocities = false, hasCorrespondence = false, hasTracking = false, hasScales = false, hasNeighbors = false;
    AdaptiveGrid3D grid;
    SurfaceMesh3D isoSurface;
    std::vector<float> contourScales;

    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(total, indexes, particles, normals, sizes, vertexes, hasVelocities, velocities, hasCorrespondence, correspondence, hasTracking, tracking, hasScales,
           scales, hasNeighbors, neighbors, grid, isoSurface, contourScales);
    }
};

PlyProperty Scalar(const char *name, PlyType type)
{
    PlyProperty p;
    p.name = name;
    p.type = type;
    return p;
}

PlyProperty List(const char *name, PlyType type)
{
    PlyProperty p = Scalar(name, type);
    p.isList = true;
    return p;
}

// Springl PLY serialization (not used by the cached state parts, which keep the springls in cereal).
// The springls of `p` as PLY: element vertex (arm vertexes), element face (one polygon per springl, with its
// per-springl data) and element neighbor (the arm vertexes' nearest neighbors). Kept to what common mesh
// readers (MeshLab, Blender) accept: scalar properties only besides vertex_indices, and no comments. Optional
// data (velocities, correspondence, tracking, scales, neighbors) is present exactly when its properties are.
[[maybe_unused]] bool WriteSpringlsPly(std::ostream &out, const ConstellationPart &p, std::string &error)
{
    PlyTable table;

    PlyElement &vertex = table.elements.emplace_back();
    vertex.name = "vertex";
    vertex.count = p.vertexes.size();
    vertex.properties = {Scalar("x", PlyType::Float32), Scalar("y", PlyType::Float32), Scalar("z", PlyType::Float32)};
    for (const Vec3 &v : p.vertexes) {
        for (int a = 0; a < 3; a++) vertex.properties[a].values.push_back(v[a]);
    }

    PlyElement &face = table.elements.emplace_back();
    face.name = "face";
    face.count = p.indexes.size();
    std::vector<PlyProperty> &f = face.properties;
    f.push_back(List("vertex_indices", PlyType::Int32));
    f.push_back(Scalar("springl", PlyType::UInt32));
    const auto vec3 = [&](const char *x, const char *y, const char *z, const Points3 &values) {
        const size_t first = f.size();
        f.push_back(Scalar(x, PlyType::Float32));
        f.push_back(Scalar(y, PlyType::Float32));
        f.push_back(Scalar(z, PlyType::Float32));
        for (const Vec3 &v : values) {
            for (int a = 0; a < 3; a++) f[first + a].values.push_back(v[a]);
        }
    };
    uint32_t arm = 0;
    for (size_t n = 0; n < p.indexes.size(); n++) {
        for (uint32_t k = 0; k < p.sizes[n]; k++) f[0].values.push_back(arm++);
        f[0].listStart.push_back(static_cast<uint32_t>(f[0].values.size()));
        f[1].values.push_back(p.indexes[n]);
    }
    vec3("px", "py", "pz", p.particles);
    vec3("nx", "ny", "nz", p.normals);
    if (p.hasVelocities) vec3("vx", "vy", "vz", p.velocities);
    if (p.hasCorrespondence) vec3("cx", "cy", "cz", p.correspondence);
    if (p.hasTracking) {
        f.push_back(Scalar("tracking", PlyType::Int32));
        for (int t : p.tracking) f.back().values.push_back(t);
    }
    if (p.hasScales) {
        f.push_back(Scalar("scale", PlyType::Float32));
        for (float s : p.scales) f.back().values.push_back(s);
    }

    // Nearest neighbors after the mesh, one row each (grouped by arm vertex, in order), so mesh readers
    // that do not expect lists on vertexes can skip a whole element.
    if (p.hasNeighbors) {
        PlyElement &neighbor = table.elements.emplace_back();
        neighbor.name = "neighbor";
        neighbor.properties = {Scalar("vertex", PlyType::Int32), Scalar("springl", PlyType::UInt32), Scalar("edge", PlyType::Int8),
                               Scalar("distance_sqr", PlyType::Float32)};
        for (size_t v = 0; v < p.neighbors.size(); v++) {
            for (const SpringlEdge &e : p.neighbors[v]) {
                neighbor.properties[0].values.push_back(static_cast<double>(v));
                neighbor.properties[1].values.push_back(e.springl);
                neighbor.properties[2].values.push_back(e.edge);
                neighbor.properties[3].values.push_back(e.distanceSqr);
            }
        }
        neighbor.count = neighbor.properties[0].values.size();
    }
    return WritePly(out, table, error);
}

// Reads what WriteSpringlsPly() wrote into the springl fields of `p` (all but total).
[[maybe_unused]] bool ReadSpringlsPly(std::istream &in, ConstellationPart &p, std::string &error)
{
    PlyTable table;
    if (!ReadPly(in, table, error)) return false;
    const char *kMalformed = "the springl PLY file is not one springls wrote";
    const PlyElement *vertex = table.find("vertex");
    const PlyElement *face = table.find("face");
    if (!vertex || !face) {
        error = kMalformed;
        return false;
    }
    const auto get = [&](const PlyElement *e, const char *name, bool list) {
        const PlyProperty *q = e->find(name);
        return q && q->isList == list ? q : nullptr;
    };
    const PlyProperty *x = get(vertex, "x", false), *y = get(vertex, "y", false), *z = get(vertex, "z", false);
    const PlyProperty *indices = get(face, "vertex_indices", true), *springl = get(face, "springl", false);
    const auto vec3 = [&](const char *a, const char *b, const char *c) {
        return std::array<const PlyProperty *, 3>{get(face, a, false), get(face, b, false), get(face, c, false)};
    };
    const auto particle = vec3("px", "py", "pz"), normal = vec3("nx", "ny", "nz"), velocity = vec3("vx", "vy", "vz"), correspondence = vec3("cx", "cy", "cz");
    const PlyProperty *tracking = get(face, "tracking", false), *scale = get(face, "scale", false);
    const auto all = [](const std::array<const PlyProperty *, 3> &v) { return v[0] && v[1] && v[2]; };
    p.hasVelocities = all(velocity);
    p.hasCorrespondence = all(correspondence);
    p.hasTracking = tracking != nullptr;
    p.hasScales = scale != nullptr;
    p.hasNeighbors = table.find("neighbor") != nullptr;
    if (!x || !y || !z || !indices || !springl || !all(particle) || !all(normal) || (p.hasVelocities && !all(velocity)) ||
        (p.hasCorrespondence && !all(correspondence)) || (p.hasTracking && !tracking) || (p.hasScales && !scale)) {
        error = kMalformed;
        return false;
    }
    // Nearest neighbors per arm vertex, from element neighbor.
    std::vector<std::vector<SpringlEdge>> byVertex;
    if (p.hasNeighbors) {
        const PlyElement *neighbor = table.find("neighbor");
        const PlyProperty *nVertex = neighbor ? get(neighbor, "vertex", false) : nullptr, *nSpringl = neighbor ? get(neighbor, "springl", false) : nullptr,
                          *nEdge = neighbor ? get(neighbor, "edge", false) : nullptr, *nDistance = neighbor ? get(neighbor, "distance_sqr", false) : nullptr;
        if (!nVertex || !nSpringl || !nEdge || !nDistance) {
            error = kMalformed;
            return false;
        }
        byVertex.resize(vertex->count);
        for (size_t r = 0; r < neighbor->count; r++) {
            const double v = nVertex->values[r];
            if (v < 0 || v >= static_cast<double>(vertex->count)) {
                error = kMalformed;
                return false;
            }
            byVertex[static_cast<size_t>(v)].push_back(
                SpringlEdge{static_cast<uint32_t>(nSpringl->values[r]), static_cast<int8_t>(nEdge->values[r]), static_cast<float>(nDistance->values[r])});
        }
    }
    const auto at = [](const std::array<const PlyProperty *, 3> &v, size_t row) {
        return Vec3(static_cast<float>(v[0]->values[row]), static_cast<float>(v[1]->values[row]), static_cast<float>(v[2]->values[row]));
    };
    for (size_t n = 0; n < face->count; n++) {
        const uint32_t size = static_cast<uint32_t>(indices->listSize(n));
        p.sizes.push_back(size);
        for (uint32_t k = 0; k < size; k++) {
            const double index = indices->values[indices->listStart[n] + k];
            if (index < 0 || index >= static_cast<double>(vertex->count)) {
                error = kMalformed;
                return false;
            }
            const size_t v = static_cast<size_t>(index);
            p.vertexes.emplace_back(static_cast<float>(x->values[v]), static_cast<float>(y->values[v]), static_cast<float>(z->values[v]));
            if (p.hasNeighbors) p.neighbors.push_back(byVertex[v]);
        }
        p.indexes.push_back(static_cast<uint32_t>(springl->values[n]));
        p.particles.push_back(at(particle, n));
        p.normals.push_back(at(normal, n));
        if (p.hasVelocities) p.velocities.push_back(at(velocity, n));
        if (p.hasCorrespondence) p.correspondence.push_back(at(correspondence, n));
        if (p.hasTracking) p.tracking.push_back(static_cast<int>(tracking->values[n]));
        if (p.hasScales) p.scales.push_back(static_cast<float>(scale->values[n]));
    }
    return true;
}

// Copies all[n] for each n in indexes when all has one entry per springl (total of them).
template <class T>
void Gather(const std::vector<T> &all, size_t total, const std::vector<uint32_t> &indexes, bool &has, std::vector<T> &out)
{
    has = all.size() == total;
    out.clear();
    if (!has) return;
    out.reserve(indexes.size());
    for (uint32_t n : indexes) out.push_back(all[n]);
}

const char *kNotTogether = "the springl state files do not fit together";
}  // namespace

const char *AdaptiveDweaveLevelSet3D::StatePartName(StatePart part)
{
    switch (part) {
        case StatePart::State: return "state";
        case StatePart::LevelSet: return "levelset";
        case StatePart::IsoSurface: return "isosurface";
        case StatePart::Primal: return "primal";
        case StatePart::Dual: return "dual";
    }
    return "unknown";
}

const char *AdaptiveDweaveLevelSet3D::StatePartExtension(StatePart part) { return part == StatePart::LevelSet ? "vdb" : "cereal"; }

bool AdaptiveDweaveLevelSet3D::savePart(StatePart part, std::ostream &out, const std::string &source, std::string &error) const
{
    if (part == StatePart::LevelSet) {
        // The header fields as VDB metadata; loadParts() checks them as it does the cereal headers.
        const std::vector<std::pair<std::string, std::string>> header = {
            {"magic", kPartMagic}, {"version", std::to_string(kPartVersion)},          {"part", StatePartName(part)}, {"source", source},
            {"method", mName},     {"element", ElementTypeName(mManifold.elementType)}};
        if (!mGrid.saveVdb(out, header, error)) {
            error = std::string("could not write the ") + StatePartName(part) + " state: " + error;
            return false;
        }
        return true;
    }
    try {
        cereal::PortableBinaryOutputArchive ar(out);
        ar(std::string(kPartMagic), kPartVersion, std::string(StatePartName(part)), source, mName, mManifold.elementType);
        const Manifold3D &m = mManifold;
        const size_t N = m.springlCount();
        switch (part) {
            case StatePart::State:
                ar(static_cast<uint64_t>(N), mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight, mBand, mUnsignedLevelSet, mScaledSpringls,
                   mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities, mOldVertexSpringl);
                ar(mContourScales, mSplitOnRefine, mActiveConstellation, mOldConstellations);
                break;
            case StatePart::LevelSet: break;  // written above
            case StatePart::IsoSurface: ar(m.isoSurface); break;
            case StatePart::Primal:
            case StatePart::Dual: {
                const uint8_t c = static_cast<uint8_t>(part == StatePart::Primal ? Constellation::Primal : Constellation::Dual);
                ConstellationPart p;
                p.total = N;
                for (size_t n = 0; n < N; n++) {
                    if (n < m.springlConstellations.size() && m.springlConstellations[n] == c) p.indexes.push_back(static_cast<uint32_t>(n));
                }
                bool has = false;
                Gather(m.springlParticles, N, p.indexes, has, p.particles);
                Gather(m.springlParticleNormals, N, p.indexes, has, p.normals);
                Gather(m.velocities, N, p.indexes, p.hasVelocities, p.velocities);
                Gather(m.correspondence, N, p.indexes, p.hasCorrespondence, p.correspondence);
                Gather(m.springlParticleTracking, N, p.indexes, p.hasTracking, p.tracking);
                Gather(mSpringlScales, N, p.indexes, p.hasScales, p.scales);
                p.hasNeighbors = mNearestNeighbors.size() == m.springlVertexes.size();
                for (uint32_t n : p.indexes) {
                    p.sizes.push_back(static_cast<uint32_t>(m.springlSize(n)));
                    for (uint32_t v = m.springlBegin(n); v < m.springlStart[n + 1]; v++) {
                        p.vertexes.push_back(m.springlVertexes[v]);
                        if (p.hasNeighbors) p.neighbors.push_back(mNearestNeighbors[v]);
                    }
                }
                p.grid = mStates[c].grid;
                p.isoSurface = mStates[c].isoSurface;
                p.contourScales = mStates[c].contourScales;
                ar(p);
                break;
            }
        }
    } catch (const std::exception &e) {
        error = std::string("could not write the ") + StatePartName(part) + " state: " + e.what();
        return false;
    }
    if (!out) {
        error = std::string("could not write the ") + StatePartName(part) + " state";
        return false;
    }
    return true;
}

bool AdaptiveDweaveLevelSet3D::loadParts(const std::array<std::istream *, kStatePartCount> &in, const std::string &source, std::string &error)
{
    const auto stream = [&](StatePart part) -> std::istream & { return *in[static_cast<size_t>(part)]; };
    // Everything is read into a copy and moved in at the end, so a failure leaves this simulation as it was.
    AdaptiveDweaveLevelSet3D next(*this);
    std::array<ConstellationPart, kConstellationCount> parts;
    uint64_t total = 0;
    const auto checkHeader = [&](StatePart part, const std::string &magic, uint32_t version, const std::string &name, const std::string &tag,
                                 const std::string &method, ElementType type) {
        if (magic != kPartMagic || version != kPartVersion || name != StatePartName(part)) {
            error = std::string("not a springls ") + StatePartName(part) + " state file (or a different version)";
            return false;
        }
        if (tag != source) {
            error = std::string("the ") + StatePartName(part) + " state is of '" + tag + "', not '" + source + "'";
            return false;
        }
        if (method != mName || type != mManifold.elementType) {
            error = std::string("the ") + StatePartName(part) + " state is of " + method + " with " + ElementTypeName(type) + " springls, not " + mName +
                    " with " + ElementTypeName(mManifold.elementType);
            return false;
        }
        return true;
    };
    const auto readHeader = [&](cereal::PortableBinaryInputArchive &ar, StatePart part) {
        std::string magic, name, tag, method;
        uint32_t version = 0;
        ElementType type = ElementType::Quad;
        ar(magic, version, name, tag, method, type);
        return checkHeader(part, magic, version, name, tag, method, type);
    };
    {
        std::vector<std::pair<std::string, std::string>> meta;
        std::string vdbError;
        if (!next.mGrid.loadVdb(stream(StatePart::LevelSet), meta, vdbError)) {
            error = std::string("could not read the ") + StatePartName(StatePart::LevelSet) + " state: " + vdbError;
            return false;
        }
        std::map<std::string, std::string> header(meta.begin(), meta.end());
        const std::string &element = header["element"];
        const bool quad = element == ElementTypeName(ElementType::Quad);
        uint32_t version = 0;  // also for an unknown element type
        if (quad || element == ElementTypeName(ElementType::Triangle)) {
            try {
                version = static_cast<uint32_t>(std::stoul(header["version"]));
            } catch (const std::exception &) {
            }
        }
        if (!checkHeader(StatePart::LevelSet, header["magic"], version, header["part"], header["source"], header["method"],
                         quad ? ElementType::Quad : ElementType::Triangle)) {
            return false;
        }
    }
    try {
        {
            cereal::PortableBinaryInputArchive ar(stream(StatePart::State));
            if (!readHeader(ar, StatePart::State)) return false;
            ar(total, next.mSimulationTime, next.mIteration, next.mAdvectionSubstepCount, next.mAdvectionWeight, next.mBand, next.mUnsignedLevelSet,
               next.mScaledSpringls, next.mOldVertexes, next.mOldParticles, next.mOldNormals, next.mOldCorrespondences, next.mOldVelocities,
               next.mOldVertexSpringl);
            ar(next.mContourScales, next.mSplitOnRefine, next.mActiveConstellation, next.mOldConstellations);
        }
        {
            cereal::PortableBinaryInputArchive ar(stream(StatePart::IsoSurface));
            if (!readHeader(ar, StatePart::IsoSurface)) return false;
            ar(next.mManifold.isoSurface);
        }
        const StatePart names[kConstellationCount] = {StatePart::Primal, StatePart::Dual};
        for (size_t c = 0; c < kConstellationCount; c++) {
            cereal::PortableBinaryInputArchive ar(stream(names[c]));
            if (!readHeader(ar, names[c])) return false;
            ar(parts[c]);
        }
    } catch (const std::exception &e) {
        error = std::string("could not read the state: ") + e.what();
        return false;
    }

    // The constellations must partition the springls, with arrays that agree.
    const ConstellationPart &first = parts[0];
    std::vector<int8_t> owner(total, -1);
    std::vector<uint32_t> slot(total, 0);
    std::array<std::vector<size_t>, kConstellationCount> armStart;
    for (size_t c = 0; c < kConstellationCount; c++) {
        const ConstellationPart &p = parts[c];
        const size_t n = p.indexes.size();
        armStart[c].assign(1, 0);
        for (uint32_t s : p.sizes) armStart[c].push_back(armStart[c].back() + s);
        const size_t arms = armStart[c].back();
        const bool consistent = p.total == total && p.particles.size() == n && p.normals.size() == n && p.sizes.size() == n && p.vertexes.size() == arms &&
                                (!p.hasVelocities || p.velocities.size() == n) && (!p.hasCorrespondence || p.correspondence.size() == n) &&
                                (!p.hasTracking || p.tracking.size() == n) && (!p.hasScales || p.scales.size() == n) &&
                                (!p.hasNeighbors || p.neighbors.size() == arms) && p.hasVelocities == first.hasVelocities &&
                                p.hasCorrespondence == first.hasCorrespondence && p.hasTracking == first.hasTracking && p.hasScales == first.hasScales &&
                                p.hasNeighbors == first.hasNeighbors;
        if (!consistent) {
            error = kNotTogether;
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            const uint32_t g = p.indexes[i];
            if (g >= total || owner[g] >= 0 || p.sizes[i] < 3 || p.sizes[i] > static_cast<uint32_t>(kMaxSpringlSize)) {
                error = kNotTogether;
                return false;
            }
            owner[g] = static_cast<int8_t>(c);
            slot[g] = static_cast<uint32_t>(i);
        }
    }
    if (std::find(owner.begin(), owner.end(), int8_t(-1)) != owner.end()) {
        error = kNotTogether;
        return false;
    }

    // Merge in global order.
    Manifold3D &m = next.mManifold;
    m.springlParticles.clear();
    m.springlParticleNormals.clear();
    m.velocities.clear();
    m.correspondence.clear();
    m.springlParticleTracking.clear();
    m.springlConstellations.clear();
    m.clearSpringlVertexes();
    next.mSpringlScales.clear();
    next.mNearestNeighbors.clear();
    for (size_t g = 0; g < total; g++) {
        const size_t c = static_cast<size_t>(owner[g]);
        const ConstellationPart &p = parts[c];
        const uint32_t i = slot[g];
        m.springlParticles.push_back(p.particles[i]);
        m.springlParticleNormals.push_back(p.normals[i]);
        if (first.hasVelocities) m.velocities.push_back(p.velocities[i]);
        if (first.hasCorrespondence) m.correspondence.push_back(p.correspondence[i]);
        if (first.hasTracking) m.springlParticleTracking.push_back(p.tracking[i]);
        if (first.hasScales) next.mSpringlScales.push_back(p.scales[i]);
        m.springlConstellations.push_back(static_cast<uint8_t>(c));
        const size_t a = armStart[c][i], b = armStart[c][i + 1];
        m.appendSpringlVertexes(std::span<const Vec3>(p.vertexes.data() + a, b - a));
        if (first.hasNeighbors) next.mNearestNeighbors.insert(next.mNearestNeighbors.end(), p.neighbors.begin() + a, p.neighbors.begin() + b);
    }
    for (size_t c = 0; c < kConstellationCount; c++) {
        next.mStates[c].grid = std::move(parts[c].grid);
        next.mStates[c].isoSurface = std::move(parts[c].isoSurface);
        next.mStates[c].contourScales = std::move(parts[c].contourScales);
    }
    next.mSpringlGrid.invalidate();
    next.mFillCandidatesStamp = std::numeric_limits<uint64_t>::max();
    *this = std::move(next);
    return true;
}

}  // namespace springls
