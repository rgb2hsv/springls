// Checkpoints of the 3D simulations (SpringLevelSet3D::saveState / loadState), written with cereal's
// portable binary archive. Every class writes its own members after its base's (writeState /
// readState); value types (Volume, NarrowBand3D, SurfaceMesh3D, Manifold3D, AdaptiveGrid3D, ...)
// carry cereal serialize / save / load member templates in their headers.
#include <springls/AdaptiveSpringLevelSet3D.hpp>
#include <springls/DoubleweaveLevelSet3D.hpp>
#include <springls/Vdb.hpp>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <exception>
#include <istream>
#include <ostream>

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
        !header->metadata("springls.method", method) || !header->metadata("springls.element", element) ||
        !header->metadata("springls.iteration", iteration) || magic != kMagic || version != static_cast<int32_t>(kVersion)) {
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
    const_cast<NarrowBand3D &>(mBand).serializeBand(ar);
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

void DoubleweaveLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet3D::writeState(ar);
    for (const ConstellationLevelSet &c : mConstellations) {
        const_cast<NarrowBand3D &>(c.band).serializeBand(ar);
        ar(c.isoSurface, c.valid);
    }
    ar(mActiveConstellation, mOldConstellations);
}

void DoubleweaveLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet3D::readState(ar);
    for (ConstellationLevelSet &c : mConstellations) {
        c.band.serializeBand(ar);
        ar(c.isoSurface, c.valid);
    }
    ar(mActiveConstellation, mOldConstellations);
}

bool DoubleweaveLevelSet3D::writeLevelSets(VdbWriter &writer, std::string &error) const
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

bool DoubleweaveLevelSet3D::readLevelSets(const VdbReader &reader, std::string &error)
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

void AdaptiveDoubleweaveLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    AdaptiveSpringLevelSet3D::writeState(ar);
    for (const ConstellationState &c : mStates) ar(c.isoSurface, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

bool AdaptiveDoubleweaveLevelSet3D::writeLevelSets(VdbWriter &writer, std::string &error) const
{
    if (!AdaptiveSpringLevelSet3D::writeLevelSets(writer, error)) return false;
    for (size_t c = 0; c < kConstellationCount; c++) {
        if (!WriteGrid(writer, mStates[c].grid, kConstellationPrefix[c], error)) return false;
    }
    return true;
}

bool AdaptiveDoubleweaveLevelSet3D::readLevelSets(const VdbReader &reader, std::string &error)
{
    if (!AdaptiveSpringLevelSet3D::readLevelSets(reader, error)) return false;
    for (size_t c = 0; c < kConstellationCount; c++) {
        if (!ReadGrid(reader, mStates[c].grid, kConstellationPrefix[c], error)) return false;
    }
    return true;
}

void AdaptiveDoubleweaveLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    AdaptiveSpringLevelSet3D::readState(ar);
    for (ConstellationState &c : mStates) ar(c.isoSurface, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

}  // namespace springls
