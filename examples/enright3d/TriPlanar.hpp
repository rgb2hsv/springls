#pragma once
#include "Experiment.hpp"

#include <array>
#include <vector>

namespace enright3d
{
// Three axis-aligned slices of a run's level-set grid, drawn into ImGui: the plane at a fixed X
// (Y right, Z up), at a fixed Y (X right, Z up) and at a fixed Z (X right, Y up), each with a
// slider below it to move the slice. Each pixel is colored by the octree depth of the leaf it
// falls in (depth 0 for uniform grids), block outlines are darker, the inside of the surface is
// lighter, and the surface's exact cross-section is drawn in white. Needs a current OpenGL 3 context (the slices are textures).
class TriPlanarView
{
public:
    static constexpr int kResolution = 256;  // pixels per slice side

    ~TriPlanarView();
    // Draws the three slices of `run`, sized to fit the remaining space of the current ImGui window
    // without scrolling.
    void draw(const ExperimentSnapshot &snapshot, const RunSnapshot &run);

private:
    struct Plane {
        unsigned int texture = 0;
        float slice = -1.0f;  // voxels along the plane's normal axis (< 0: not set yet)
        // What the texture shows, to regenerate only on change.
        uint64_t generation = 0, iteration = 0;
        float shownSlice = -1.0f;
        const void *shownRun = nullptr;
    };
    std::array<Plane, 3> mPlanes;
    std::vector<unsigned char> mPixels;

    void render(const ExperimentSnapshot &snapshot, const RunSnapshot &run, int axis);
};
}  // namespace enright3d
