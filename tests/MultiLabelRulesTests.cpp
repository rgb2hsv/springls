// Label-aware springl rules in SpringLevelSet2D: springls of one label never attract, fill against
// or survive on account of springls of another label.
#include "Check.hpp"

#include <springl/SpringLevelSet2D.hpp>

using namespace springls;

namespace
{
// Exposes the extension points and lets a test hand-build the manifold.
struct LabelProbe : SpringLevelSet2D {
    explicit LabelProbe(bool labeled) { mLabeled = labeled; }
    using SpringLevelSet2D::appendSpringl;
    using SpringLevelSet2D::compactSpringls;
    using SpringLevelSet2D::mManifold;
    using SpringLevelSet2D::mNearestNeighbor;
    using SpringLevelSet2D::updateNearestNeighbors;
    // The hand-built contour stands in for the level set's.
    void solveContourForFill() override {}
    int fillGaps() { return fill(); }
};

void AddSpringl(LabelProbe &probe, Vec2 a, Vec2 b, int label) { probe.appendSpringl(0.5f * (a + b), a, b, 1.0f, label); }
}  // namespace

TEST(labeled_springls_ignore_other_labels)
{
    LabelProbe probe(true);
    AddSpringl(probe, Vec2(5, 5), Vec2(6, 5), 1);
    AddSpringl(probe, Vec2(5, 5), Vec2(6, 5), 2);
    AddSpringl(probe, Vec2(6.05f, 5), Vec2(7, 5), 1);
    probe.mManifold.updateNormals();
    probe.updateNearestNeighbors(0.5f);
    // Springl 0's end vertex (6, 5) reaches the label-1 start vertex at (6.05, 5) (index 4), not
    // the coincident label-2 vertex.
    CHECK(probe.mNearestNeighbor[1] == 4);
    // The label-2 springl has no same-label neighbor.
    CHECK(probe.mNearestNeighbor[2] == -1);
    CHECK(probe.mNearestNeighbor[3] == -1);
}

TEST(compact_keeps_labels_in_sync)
{
    LabelProbe probe(true);
    AddSpringl(probe, Vec2(1, 1), Vec2(2, 1), 1);
    AddSpringl(probe, Vec2(3, 1), Vec2(4, 1), 2);
    AddSpringl(probe, Vec2(5, 1), Vec2(6, 1), 3);
    probe.compactSpringls({1, 0, 1});
    CHECK(probe.mManifold.springlLabels.size() == 2);
    CHECK(probe.mManifold.springlLabels[0] == 1);
    CHECK(probe.mManifold.springlLabels[1] == 3);
}

TEST(fill_uses_curve_label)
{
    LabelProbe probe(true);
    Manifold2D &m = probe.mManifold;
    m.vertexLocations = {Vec2(5, 5), Vec2(6, 5)};
    m.vertexIndexes = {{0, 1}};
    m.curveLabels = {2};
    CHECK(probe.fillGaps() == 1);
    CHECK(m.springlLabels.size() == 1 && m.springlLabels.back() == 2);

    // A springl of another label on the same segment does not close the gap.
    LabelProbe other(true);
    other.mManifold.vertexLocations = m.vertexLocations;
    other.mManifold.vertexIndexes = m.vertexIndexes;
    other.mManifold.curveLabels = {2};
    AddSpringl(other, Vec2(5, 5), Vec2(6, 5), 1);
    CHECK(other.fillGaps() == 1);
    CHECK(other.mManifold.springlLabels.back() == 2);
}

TEST(unlabeled_labels_stay_empty)
{
    LabelProbe probe(false);
    AddSpringl(probe, Vec2(1, 1), Vec2(2, 1), 0);
    CHECK(probe.mManifold.springlLabels.empty());
    CHECK(probe.springlLabel(0) == SpringLevelSet2D::kBackgroundLabel);
}
