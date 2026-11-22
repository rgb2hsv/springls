#pragma once
#include <mesh/MeshDistance3D.hpp>
#include <spatial/AdaptiveGrid3D.hpp>
#include <springl/SpringLevelSet3D.hpp>

#include <array>
#include <functional>
#include <memory>

namespace springls
{
// Spring level set on a block-adaptive octree (AdaptiveGrid3D).
//
// The octree stores a narrow band at a single, finest resolution around the springls, as an
// OpenVDB grid does: before each level-set update, every leaf within the band margin (see bandCells)
// of a springl particle is refined to adaptiveTuning.maxDepth, and parents with no particle within
// the margin coarsen (one depth per update). The margin covers how far springls move in a
// sub-step plus the active band, so the zero level set always lies in finest leaves; coarser
// leaves only fill the space away from the surface. At init, the band follows the initial
// interface instead. Springls are created at the finest cell size; any springl that a refinement
// leaves larger than its leaf is split (see splitSpringl()).
class AdaptiveSpringLevelSet3D : public SpringLevelSet3D
{
public:
    struct AdaptiveTuning {
        int maxDepth = 3;        // band cells are 2^-maxDepth base voxels
        float bandWidth = 3.0f;  // phi is clamped to +-bandWidth (base voxels)
        // Leaves within bandCells finest cells plus one sub-step of motion (tuning.maxStepVoxels) of a
        // springl particle are at maxDepth: enough for the zero layer, its three rebuilt distance
        // layers and a cell of slack to stay in finest leaves until the band is re-fitted, at the
        // next sub-step.
        float bandCells = 5.0f;
        // Active narrow band: leaves within activeCells of their own cells, plus activeMargin base
        // voxels (about one sub-step of motion), of the interface are evolved.
        float activeCells = 3.0f;
        float activeMargin = 0.35f;
        // After each evolution, corners within a cell of a springl that lie on the wrong side of it
        // take its signed distance (particle-level-set style one-sided correction).
        bool springlCorrection = true;
    };
    AdaptiveTuning adaptiveTuning;

    explicit AdaptiveSpringLevelSet3D(ElementType elementType = ElementType::Quad);
    bool step() override;
    // Initial interface as a closed mesh in grid space (base voxels), splatted into the octree at init()
    // (SplatMeshIntoGrid) for exact distances at the finest resolution; the initial level set then only sets
    // the domain size. Null (the default) refines from the initial level set instead.
    void setInitialSurface(std::shared_ptr<const MeshDistance3D> gridSpaceMesh) { mInitialSurface = std::move(gridSpaceMesh); }

    const AdaptiveGrid3D &getGrid() const { return mGrid; }
    // Largest |phi| at a springl particle, in local cell sizes (a diagnostic).
    float maxTrackingError() const;

protected:
    AdaptiveGrid3D mGrid;
    std::vector<float> mContourScales;
    std::shared_ptr<const MeshDistance3D> mInitialSurface;
    // Whether refining mGrid splits springls that no longer fit their leaf.
    bool mSplitOnRefine = true;
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    bool writeLevelSets(VdbWriter &writer, std::string &error) const override;
    bool readLevelSets(const VdbReader &reader, std::string &error) override;

    void initLevelSet() override;
    float contourScale(uint32_t vertex) const override { return (vertex < mContourScales.size()) ? mContourScales[vertex] : 1.0f; }
    // Attraction reaches as far as the level set is resolved where the vertex currently is.
    float attractionScale(const Vec3 &position, size_t) const override { return mGrid.cellSize(position); }
    Vec3 levelSetGradient(const Vec3 &p) const override;
    void evolveLevelSetAndContour() override;

    // Refines leaves for which near(block) holds to maxDepth, then coarsens (one depth) parents
    // none of whose children are near.
    void adaptBand(const std::function<bool(const AdaptiveGrid3D::Block &)> &near);
    // The band around the particles of the springls that drive mGrid.
    void adaptToParticles();
    // The band around `points`: leaves within bandMargin() of one are refined to maxDepth.
    void adaptToPoints(const Points3 &points);
    // Distance (base voxels) within which leaves are kept at maxDepth.
    virtual float bandMargin() const { return tuning.maxStepVoxels + adaptiveTuning.bandCells * std::ldexp(1.0f, -adaptiveTuning.maxDepth); }
    // Extracts the iso-surface (normals oriented outward).
    void extractGridSurface();
    // Sparse-field evolution toward the springls, the springl correction, then iso-surface extraction.
    virtual void evolveGrid();
    void splitOversizedSpringls();
    // Appends the half-size children of springl `index` (the caller removes the parent): four
    // midpoint triangles for a triangle, one quad per corner for any other polygon.
    virtual void splitSpringl(size_t index);
    // Springls whose particles and elements drive mGrid (default: all).
    virtual bool levelSetIncludes(size_t) const { return true; }
};

// DoubleWeave springls on a block-adaptive octree: the combined grid (all springls) is the output
// surface and decides springl splitting; each constellation also keeps its own grid, banded
// around and evolved from its own springls, against which it is filled and contracted. Dual
// springls come from ForEachDualSpringl, as in DweaveLevelSet3D.
class AdaptiveDweaveLevelSet3D : public AdaptiveSpringLevelSet3D
{
public:
    explicit AdaptiveDweaveLevelSet3D(ElementType elementType = ElementType::Quad);
    bool init() override;
    size_t countSpringls(Constellation constellation) const;

    // The state split into separate files, e.g. to cache the state after init(): the scalar simulation
    // state, the combined adaptive level set as an OpenVDB file (AdaptiveGrid3D::saveVdb), its iso-surface,
    // and each constellation's springls with that constellation's own level set; all but the level set are
    // cereal portable-binary files. Each part is tagged with `source` (text naming where the state came
    // from, such as the source mesh and settings); loadParts() refuses parts whose tag, method or element
    // type differ, or that do not fit together. Configure the simulation as for init() (tuning, initial
    // level set); loadParts() then replaces init(). Springl order is preserved, so a restored simulation
    // steps exactly as the saved one would. Return false with a message on failure; a failed loadParts()
    // leaves the simulation unchanged.
    enum class StatePart {
        State,
        LevelSet,
        IsoSurface,
        Primal,
        Dual
    };
    static constexpr size_t kStatePartCount = 5;
    static constexpr std::array<StatePart, kStatePartCount> kStateParts = {StatePart::State, StatePart::LevelSet, StatePart::IsoSurface, StatePart::Primal,
                                                                           StatePart::Dual};
    static const char *StatePartName(StatePart part);       // "state", "levelset", "isosurface", "primal", "dual"
    static const char *StatePartExtension(StatePart part);  // "vdb" for the level set, "cereal" for the others
    bool savePart(StatePart part, std::ostream &out, const std::string &source, std::string &error) const;
    // parts[i] is the stream of kStateParts[i].
    bool loadParts(const std::array<std::istream *, kStatePartCount> &parts, const std::string &source, std::string &error);

protected:
    struct ConstellationState {
        AdaptiveGrid3D grid;
        SurfaceMesh3D isoSurface;
        std::vector<float> contourScales;
    };
    std::array<ConstellationState, kConstellationCount> mStates;
    int mActiveConstellation = -1;
    std::vector<uint8_t> mOldConstellations;
    // Fill candidates per constellation (springls its iso-surface would seed), shared by the fill
    // passes of one sub-step; mFillCandidatesStamp is the sub-step they were built in.
    struct FillCandidate {
        Vec3 particle;
        std::array<Vec3, kMaxSpringlSize> vertexes;
        int size = 0;
        float scale = 1.0f;
    };
    std::array<std::vector<FillCandidate>, kConstellationCount> mFillCandidates;
    uint64_t mFillCandidatesStamp = std::numeric_limits<uint64_t>::max();
    void writeState(cereal::PortableBinaryOutputArchive &ar) const override;
    void readState(cereal::PortableBinaryInputArchive &ar) override;
    bool writeLevelSets(VdbWriter &writer, std::string &error) const override;
    bool readLevelSets(const VdbReader &reader, std::string &error) override;

    void swapState(size_t c);
    void evolveConstellations();
    size_t appendSpringl(const Vec3 &particle, std::span<const Vec3> vertexes, float scale, Constellation c);

    void seedSpringls() override;
    void clearSpringls() override;
    void compactSpringls(const std::vector<uint8_t> &keep) override;
    void snapshotForTracking() override;
    bool canTrackFrom(size_t newIndex, size_t oldIndex) const override;
    bool canAttract(size_t springl, size_t other) const override { return mManifold.springlConstellations[springl] == mManifold.springlConstellations[other]; }
    int fill() override;
    int contract() override;
    void evolveLevelSetAndContour() override;
    bool refinesContourWith(size_t particle) const override { return levelSetIncludes(particle); }
    bool levelSetIncludes(size_t springl) const override;
    void splitSpringl(size_t index) override;
};
}  // namespace springls
