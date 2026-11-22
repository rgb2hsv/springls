#include <springl/MultiSpringLevelSet2D.hpp>

#include <levelset/Contour2D.hpp>
#include <springl/SpringlRegionProbe2D.hpp>

namespace springls
{
MultiSpringLevelSet2D::MultiSpringLevelSet2D()
{
    mName = "Multi-Object Springls";
    mLabeled = true;
}

void MultiSpringLevelSet2D::initLevelSet()
{
    mMultiBand.initialize(mInitialLabels, mInitialLabelDistance);
    mObjectLabels = ObjectLabels(mInitialLabels);
    // The crossings interpolate exact initial distances, so the initial contour stays where it is;
    // its spacing is evened out as the single-object one is (springls are seeded per segment).
    extractContour();
    evenInitialContourSpacing();
}

void MultiSpringLevelSet2D::extractContour()
{
    const MultiNarrowBand2D &band = mMultiBand;
    ExtractMultiLabelContour([&](int i, int j, int l) { return band.signedValue(i, j, l); }, band.labels().width(), band.labels().height(), mObjectLabels,
                             mManifold.vertexLocations, mManifold.vertexIndexes, mManifold.curveLabels, 1E-3f, &mContourJunctions);
}

void MultiSpringLevelSet2D::updateMultiLevelSet(float maxStep)
{
    const std::vector<Vec2i> &active = mMultiBand.active();
    const auto &candidates = mMultiBand.candidates();
    auto &delta = mMultiBand.delta();
    const int N = static_cast<int>(active.size());
    SPRINGLS_OMP(omp parallel for if (N >= kParallelMinHeavyItems))
    for (int n = 0; n < N; n++) {
        const Vec2i p = active[n];
        if (mMultiBand.previousDistance()(p) > 0.5f) continue;  // only the zero layer moves
        const Vec2 pull = unsignedPull(p.x(), p.y());
        for (int c = 0; c < MultiNarrowBand2D::kCandidates; c++) {
            const int label = candidates[n][c];
            if (label == MultiNarrowBand2D::kNoLabel) continue;
            if (label == kBackgroundLabel) {
                delta[n][c] = mMultiBand.previousCurvatureTerm(p.x(), p.y(), label, backgroundCurvatureWeight);
            } else {
                delta[n][c] =
                    -mMultiBand.previousUpwindAdvection(p.x(), p.y(), label, pull) + mMultiBand.previousCurvatureTerm(p.x(), p.y(), label, mCurvatureWeight);
            }
        }
    }
    mMultiBand.update(maxStep);
}

void MultiSpringLevelSet2D::evolveLevelSetAndContour()
{
    updateUnsignedLevelSet();
    for (int i = 0; i < tuning.signedEvolveIterations; i++) updateMultiLevelSet();
    if (tuning.springlCorrection) correctFromSpringls();
    afterLevelSetUpdate();
    extractContour();
    refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
    mManifold.updateNormals();
}

void MultiSpringLevelSet2D::correctFromSpringls(const std::function<bool(size_t)> &include)
{
    // A band pixel within a cell of a springl, beside it and on its inner side, lies in that
    // springl's region (one-sided: the region beyond a springl is unknown).
    mManifold.updateNormals();
    const SpringlRegionProbe2D probe(mManifold, include);
    mMultiBand.correctLabels(
        [&](int i, int j, int &label, float &d) { return probe.inside(Vec2(static_cast<float>(i), static_cast<float>(j)), 1.0f, label, d); });
}

void MultiSpringLevelSet2D::solveContourForFill()
{
    extractContour();
    refineContour(tuning.refineIterations, tuning.refineProximity, tuning.refineStepSize);
}
}  // namespace springls
