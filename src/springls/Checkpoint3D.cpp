// Checkpoints of the 3D simulations (SpringLevelSet3D::saveState / loadState), written with cereal's
// portable binary archive. Every class writes its own members after its base's (writeState /
// readState); value types (Volume, NarrowBand3D, SurfaceMesh3D, Manifold3D, AdaptiveGrid3D, ...)
// carry cereal serialize / save / load member templates in their headers.
#include <springls/AdaptiveSpringLevelSet3D.hpp>
#include <springls/DoubleweaveLevelSet3D.hpp>

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
constexpr uint32_t kVersion = 2;  // 2: the semi-Lagrangian rebuild times are no longer stored
const char *ElementTypeName(ElementType type) { return type == ElementType::Quad ? "quad" : "triangle"; }
}  // namespace

bool SpringLevelSet3D::saveState(std::ostream &out, std::string &error) const
{
    try {
        cereal::PortableBinaryOutputArchive ar(out);
        ar(std::string(kMagic), kVersion, mName, mManifold.elementType);
        writeState(ar);
    } catch (const std::exception &e) {
        error = std::string("could not write the state: ") + e.what();
        return false;
    }
    if (!out) {
        error = "could not write the state";
        return false;
    }
    return true;
}

bool SpringLevelSet3D::loadState(std::istream &in, std::string &error)
{
    try {
        cereal::PortableBinaryInputArchive ar(in);
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
    return true;
}

void SpringLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    ar(mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight, mBand, mUnsignedLevelSet, mManifold, mNearestNeighbors, mScaledSpringls,
       mSpringlScales, mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities, mOldVertexSpringl);
}

void SpringLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    ar(mSimulationTime, mIteration, mAdvectionSubstepCount, mAdvectionWeight, mBand, mUnsignedLevelSet, mManifold, mNearestNeighbors, mScaledSpringls,
       mSpringlScales, mOldVertexes, mOldParticles, mOldNormals, mOldCorrespondences, mOldVelocities, mOldVertexSpringl);
    mSpringlGrid.invalidate();
}

void DoubleweaveLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet3D::writeState(ar);
    for (const ConstellationLevelSet &c : mConstellations) ar(c.band, c.unsignedLevelSet, c.isoSurface, c.valid);
    ar(mActiveConstellation, mOldConstellations);
}

void DoubleweaveLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet3D::readState(ar);
    for (ConstellationLevelSet &c : mConstellations) ar(c.band, c.unsignedLevelSet, c.isoSurface, c.valid);
    ar(mActiveConstellation, mOldConstellations);
}

void AdaptiveSpringLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    SpringLevelSet3D::writeState(ar);
    ar(mGrid, mContourScales, mSplitOnRefine);
}

void AdaptiveSpringLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    SpringLevelSet3D::readState(ar);
    ar(mGrid, mContourScales, mSplitOnRefine);
}

void AdaptiveDoubleweaveLevelSet3D::writeState(cereal::PortableBinaryOutputArchive &ar) const
{
    AdaptiveSpringLevelSet3D::writeState(ar);
    for (const ConstellationState &c : mStates) ar(c.grid, c.isoSurface, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

void AdaptiveDoubleweaveLevelSet3D::readState(cereal::PortableBinaryInputArchive &ar)
{
    AdaptiveSpringLevelSet3D::readState(ar);
    for (ConstellationState &c : mStates) ar(c.grid, c.isoSurface, c.contourScales);
    ar(mActiveConstellation, mOldConstellations);
}

}  // namespace springls
