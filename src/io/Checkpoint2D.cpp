// Checkpoints of the 2D simulations (SpringLevelSet2D::saveState / loadState), written with cereal's
// portable binary archive as one stream: every class writes its own members after its base's
// (writeState / readState); value types (Image, NarrowBand2D, MultiNarrowBand2D, AdaptiveGrid2D,
// SpringlHashGrid2D) carry cereal serialize / save / load member templates in their headers. The 2D
// grids are small, so unlike the 3D checkpoints the level sets are stored in the same stream.
#include <dweave/AdaptiveMultiDweaveLevelSet2D.hpp>
#include <dweave/DweaveLevelSet2D.hpp>
#include <dweave/MultiDweaveLevelSet2D.hpp>
#include <levelset/MultiActiveContour2D.hpp>
#include <springl/AdaptiveSpringLevelSet2D.hpp>
#include <springl/MultiSpringLevelSet2D.hpp>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/common.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <exception>
#include <istream>
#include <ostream>
#include <string>

namespace Eigen
{
// Fixed-size Eigen vectors (Vec2, Vec2i) as their coefficients.
template <class Archive, class Scalar, int Rows>
void serialize(Archive &ar, Matrix<Scalar, Rows, 1> &v)
{
    for (int i = 0; i < Rows; i++) ar(v[i]);
}
}  // namespace Eigen

namespace springls
{
template <class Archive>
void serialize(Archive &ar, Manifold2D &m)
{
    ar(m.vertexLocations, m.vertexIndexes, m.springlParticles, m.springlVertexes, m.springlParticleNormals, m.springlConstellations, m.springlLabels,
       m.curveLabels, m.springlParticleTracking, m.velocities, m.correspondence);
}

namespace
{
constexpr char kMagic[] = "springls-2d-state";
constexpr uint32_t kVersion = 1;
}  // namespace

bool SpringLevelSet2D::saveState(std::ostream &state, std::string &error) const
{
    try {
        cereal::PortableBinaryOutputArchive ar(state);
        ar(std::string(kMagic), kVersion, mName);
        writeState(ar);
    } catch (const std::exception &e) {
        error = std::string("could not write the state: ") + e.what();
        return false;
    }
    if (!state) {
        error = "could not write the state";
        return false;
    }
    return true;
}

bool SpringLevelSet2D::loadState(std::istream &state, std::string &error)
{
    try {
        cereal::PortableBinaryInputArchive ar(state);
        std::string magic, name;
        uint32_t version = 0;
        ar(magic, version);
        if (magic != kMagic || version != kVersion) {
            error = "not a springls 2D state file (or a different version)";
            return false;
        }
        ar(name);
        if (name != mName) {
            error = "the state is of " + name + ", not " + mName;
            return false;
        }
        readState(ar);
    } catch (const std::exception &e) {
        error = std::string("could not read the state: ") + e.what();
        return false;
    }
    return true;
}

void SpringLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    ar(mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight);
    ar(mBand, mUnsignedLevelSet, mManifold, mContourJunctions, mNearestNeighbor, mScaledSpringls, mSpringlScales, mSpringlGrid);
    ar(mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities, mOldLabels);
}

void SpringLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    ar(mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight);
    ar(mBand, mUnsignedLevelSet, mManifold, mContourJunctions, mNearestNeighbor, mScaledSpringls, mSpringlScales, mSpringlGrid);
    ar(mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities, mOldLabels);
}

void DweaveLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet2D::writeState(ar);
    for (const ConstellationLevelSet &c : mConstellations) ar(c.band, c.unsignedLevelSet, c.vertexLocations, c.vertexIndexes, c.valid);
    ar(mActiveConstellation, mOldConstellations);
}

void DweaveLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet2D::readState(ar);
    for (ConstellationLevelSet &c : mConstellations) ar(c.band, c.unsignedLevelSet, c.vertexLocations, c.vertexIndexes, c.valid);
    ar(mActiveConstellation, mOldConstellations);
}

void MultiSpringLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet2D::writeState(ar);
    ar(mMultiBand, mObjectLabels);
}

void MultiSpringLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet2D::readState(ar);
    ar(mMultiBand, mObjectLabels);
}

void MultiDweaveLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    MultiSpringLevelSet2D::writeState(ar);
    for (const ConstellationLevelSet &c : mConstellations) ar(c.band, c.unsignedLevelSet, c.vertexLocations, c.vertexIndexes, c.curveLabels, c.valid);
    ar(mActiveConstellation, mOldConstellations);
}

void MultiDweaveLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    MultiSpringLevelSet2D::readState(ar);
    for (ConstellationLevelSet &c : mConstellations) ar(c.band, c.unsignedLevelSet, c.vertexLocations, c.vertexIndexes, c.curveLabels, c.valid);
    ar(mActiveConstellation, mOldConstellations);
}

void MultiActiveContour2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet2D::writeState(ar);
    ar(mMultiBand, mObjectLabels);
}

void MultiActiveContour2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet2D::readState(ar);
    ar(mMultiBand, mObjectLabels);
}

void AdaptiveSpringLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet2D::writeState(ar);
    ar(mGrid, mContourScales, mSplitOnRefine);
}

void AdaptiveSpringLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet2D::readState(ar);
    ar(mGrid, mContourScales, mSplitOnRefine);
}

void AdaptiveDweaveLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    AdaptiveSpringLevelSet2D::writeState(ar);
    for (const ConstellationState &c : mStates) ar(c.grid, c.vertexLocations, c.vertexIndexes, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

void AdaptiveDweaveLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    AdaptiveSpringLevelSet2D::readState(ar);
    for (ConstellationState &c : mStates) ar(c.grid, c.vertexLocations, c.vertexIndexes, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

void AdaptiveMultiDweaveLevelSet2D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    AdaptiveMultiSpringLevelSet2D::writeState(ar);
    for (const ConstellationState &c : mStates) ar(c.grid, c.vertexLocations, c.vertexIndexes, c.curveLabels, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

void AdaptiveMultiDweaveLevelSet2D::readState(cereal::PortableBinaryInputArchive &ar)
{
    AdaptiveMultiSpringLevelSet2D::readState(ar);
    for (ConstellationState &c : mStates) ar(c.grid, c.vertexLocations, c.vertexIndexes, c.curveLabels, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}
}  // namespace springls
