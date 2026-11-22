// Labeled block-adaptive quadtree: region labels + unsigned distance at corners.
#include "Check.hpp"

#include <levelset/Contour2D.hpp>
#include <spatial/AdaptiveGrid2D.hpp>

#include <cmath>
#include <set>

using namespace springls;

namespace
{
constexpr int kSize = 33;  // 4 x 4 depth-0 blocks of 8 cells

using Block = AdaptiveGrid2D::Block;

// Labels split by the line coordinate = edge along axis (0: x, 1: y): `below` under it, `above` over it.
void SplitImage(int axis, float edge, int below, int above, ImageI &labels, ImageF &distance)
{
    labels = ImageI(kSize, kSize);
    distance = ImageF(kSize, kSize);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            const float c = (axis == 0) ? static_cast<float>(i) : static_cast<float>(j);
            labels(i, j) = (c < edge) ? below : above;
            distance(i, j) = std::abs(c - edge);
        }
    }
}

template <class Fn>
void ForEachLeafCorner(const AdaptiveGrid2D &grid, Fn &&fn)
{
    for (int index : grid.leaves()) {
        const Block &block = grid.blocks()[index];
        for (int b = 0; b < AdaptiveGrid2D::kBlockNodes; b++) {
            for (int a = 0; a < AdaptiveGrid2D::kBlockNodes; a++) fn(block, block.nodePosition(a, b), Block::index(a, b));
        }
    }
}

void RefineAllTo(AdaptiveGrid2D &grid, int depth)
{
    bool refined = true;
    while (refined) {
        refined = false;
        const std::vector<int> leaves = grid.leaves();
        for (int index : leaves) {
            if (grid.blocks()[index].isLeaf() && grid.blocks()[index].key.depth < depth) refined |= grid.refine(index);
        }
    }
}
}  // namespace

TEST(label_grid_refine_coarsen_round_trip)
{
    ImageI labels(kSize, kSize, 3);
    ImageF distance(kSize, kSize, 3.0f);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 2, 3.0f);
    CHECK(grid.isLabeled());
    const int root = grid.leaves().front();
    CHECK(grid.refine(root));
    const int child = grid.blocks()[root].children[0];
    CHECK(grid.refine(child));
    grid.coarsen(child);
    grid.coarsen(root);
    int wrong = 0;
    ForEachLeafCorner(grid, [&](const Block &block, const Vec2 &, int node) {
        if (block.label(node) != 3 || std::abs(block.value(node) - 3.0f) > 1E-6f) wrong++;
    });
    CHECK(wrong == 0);
}

TEST(label_grid_refine_never_invents_labels)
{
    ImageI labels;
    ImageF distance;
    SplitImage(0, 16.3f, 1, 2, labels, distance);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 3, 3.0f);
    RefineAllTo(grid, 3);
    std::set<int> seen;
    int wrongSide = 0;
    ForEachLeafCorner(grid, [&](const Block &block, const Vec2 &p, int node) {
        seen.insert(block.label(node));
        if (std::abs(p.x() - 16.3f) > 0.3f && block.label(node) != ((p.x() < 16.3f) ? 1 : 2)) wrongSide++;
    });
    CHECK((seen == std::set<int>{1, 2}));
    CHECK(wrongSide == 0);
    // Sampling between corners picks a side, never a third label.
    const MogacResult s = grid.sampleLabel(Vec2(16.1f, 5.3f));
    CHECK(s.label == 1);
    CHECK_NEAR(s.distance, 0.2f, 0.05f);
}

TEST(label_grid_refuses_mixed_coarsen)
{
    ImageI labels;
    ImageF distance;
    SplitImage(0, 12.3f, 1, 2, labels, distance);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 2, 3.0f);
    const int parent = grid.findLeaf(Vec2(12.0f, 4.0f));  // block [8, 16] x [0, 8] straddles x = 12.3
    CHECK(grid.refine(parent));
    CHECK(!grid.canCoarsen(parent));
}

TEST(label_grid_mixed_border_copies_nearer)
{
    // Horizontal boundary y = 11.7 crossing the vertical block border x = 16, with the left block
    // refined: the fine leaf's in-between corners on x = 16 straddle two labels.
    ImageI labels;
    ImageF distance;
    SplitImage(1, 11.7f, 1, 2, labels, distance);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 2, 3.0f);
    CHECK(grid.refine(grid.findLeaf(Vec2(12.0f, 12.0f))));
    grid.synchronize();
    int bad = 0, wrongSide = 0;
    ForEachLeafCorner(grid, [&](const Block &block, const Vec2 &p, int node) {
        const int l = block.label(node);
        if (l != 1 && l != 2) bad++;
        if (std::abs(p.y() - 11.7f) > 0.25f && l != ((p.y() < 11.7f) ? 1 : 2)) wrongSide++;
    });
    CHECK(bad == 0);
    CHECK(wrongSide == 0);
    const MogacResult s = grid.sampleLabel(Vec2(16.0f, 11.5f));
    CHECK(s.label == 1);
    CHECK_NEAR(s.distance, 0.2f, 0.05f);
}

TEST(label_grid_zero_speed_stable)
{
    ImageI labels;
    ImageF distance;
    SplitImage(0, 16.3f, 1, 2, labels, distance);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 2, 3.0f);
    RefineAllTo(grid, 2);
    grid.synchronize();
    grid.updateActive();
    std::vector<int> before;
    ForEachLeafCorner(grid, [&](const Block &block, const Vec2 &, int node) { before.push_back(block.label(node)); });
    // Eigen vectors are not zeroed by default construction: fill explicitly.
    AdaptiveGrid2D::NodeVelocities still;
    still.fill(Vec2::Zero());
    const std::vector<AdaptiveGrid2D::NodeVelocities> velocities(grid.activeLeaves().size(), still);
    grid.evolveLabeled(5, 0.0f, 0.0f, velocities);
    std::vector<int> after;
    ForEachLeafCorner(grid, [&](const Block &block, const Vec2 &, int node) { after.push_back(block.label(node)); });
    CHECK(before == after);
}

TEST(label_grid_uniform_pull_moves_boundary)
{
    // Every node pulled toward +x by one cell per step: label 1 (left) advances.
    ImageI labels;
    ImageF distance;
    SplitImage(0, 16.3f, 1, 2, labels, distance);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 0, 3.0f);
    grid.updateActive();
    AdaptiveGrid2D::NodeVelocities push;
    push.fill(Vec2(1.0f, 0.0f));
    const std::vector<AdaptiveGrid2D::NodeVelocities> velocities(grid.activeLeaves().size(), push);
    grid.evolveLabeled(4, 0.0f, 0.0f, velocities);
    // Four half-cell steps: the boundary moves about two cells, from 16.3 to ~18.3.
    CHECK(grid.sampleLabel(Vec2(17.5f, 10.0f)).label == 1);
    CHECK(grid.sampleLabel(Vec2(19.5f, 10.0f)).label == 2);
}

TEST(label_grid_contour_matches_fixed)
{
    ImageI labels(kSize, kSize, 0);
    for (int j = 6; j < 26; j++) {
        for (int i = 6; i < 26; i++) labels(i, j) = (i < 16) ? 1 : 2;
    }
    ImageF distance(kSize, kSize, 1.5f);
    for (int j = 0; j < kSize; j++) {
        for (int i = 0; i < kSize; i++) {
            const int l = labels(i, j);
            if (labels.at(i - 1, j) != l || labels.at(i + 1, j) != l || labels.at(i, j - 1) != l || labels.at(i, j + 1) != l) distance(i, j) = 0.5f;
        }
    }
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 0, 3.0f);
    Points2 vertices, fixedVertices;
    Curves curves, fixedCurves;
    std::vector<int> curveLabels, fixedLabels;
    std::vector<float> scales;
    grid.extractLabeledContour(vertices, curves, curveLabels, scales);
    ExtractMultiLabelContour([&](int i, int j, int l) { return (labels(i, j) == l) ? -distance(i, j) : distance(i, j); }, kSize, kSize, {1, 2}, fixedVertices,
                             fixedCurves, fixedLabels);
    CHECK(curves.size() == fixedCurves.size());
    CHECK(curveLabels == fixedLabels);
    CHECK(vertices.size() == fixedVertices.size());
    CHECK(scales.size() == vertices.size());
}

TEST(label_grid_correction_relabels_inside)
{
    ImageI labels;
    ImageF distance;
    SplitImage(0, 16.3f, 1, 2, labels, distance);
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 0, 3.0f);
    grid.updateActive();
    // A label-1 "springl" says the corner at (17, 10) lies 0.4 inside region 1.
    const int changed = grid.correctLabels([](const Vec2 &p, float, int &label, float &d) {
        if ((p - Vec2(17.0f, 10.0f)).norm() > 1E-4f) return false;
        label = 1;
        d = 0.4f;
        return true;
    });
    CHECK(changed == 1);
    CHECK(grid.sampleLabel(Vec2(17.0f, 10.0f)).label == 1);
}

TEST(label_grid_plugs_corner_on_domain_edge)
{
    // A single label-2 corner on the domain border x = 0, inside region 1: out-of-domain ghosts
    // replicate the corner itself and must not count as a same-label neighbor.
    ImageI labels(kSize, kSize, 1);
    ImageF distance(kSize, kSize, 3.0f);
    labels(0, 10) = 2;
    distance(0, 10) = distance(1, 10) = distance(0, 9) = distance(0, 11) = 0.5f;
    AdaptiveGrid2D grid;
    grid.initializeLabeled(labels, distance, 0, 3.0f);
    grid.updateActive();
    AdaptiveGrid2D::NodeVelocities still;
    still.fill(Vec2::Zero());
    grid.evolveLabeled(1, 0.0f, 0.0f, std::vector<AdaptiveGrid2D::NodeVelocities>(grid.activeLeaves().size(), still));
    CHECK(grid.sampleLabel(Vec2(0.0f, 10.0f)).label == 1);
    CHECK(grid.sampleLabel(Vec2(1.0f, 10.0f)).distance > 0.5f);
}
