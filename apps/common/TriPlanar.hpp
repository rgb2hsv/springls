#pragma once
#include <core/Volume.hpp>
#include <spatial/AdaptiveGrid3D.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace springls_gui
{
// What the slice view draws. `grid` (octree) or `levelSet` (uniform grid) gives the level set,
// `isoSurface` its exact cross-section; all lengths are in voxels.
struct TriPlanarSource {
    std::shared_ptr<const springls::AdaptiveGrid3D> grid;
    std::shared_ptr<const springls::VolumeF> levelSet;
    const springls::SurfaceMesh3D *isoSurface = nullptr;
    springls::Vec3 extent = springls::Vec3::Zero();        // domain size per axis
    springls::Vec3 defaultSlice = springls::Vec3::Zero();  // initial slice position per axis
    int maxDepth = 0;                                      // deepest octree level (legend)
    springls::Vec3i baseCells = springls::Vec3i::Zero();   // cells per axis at depth 0 (legend)
    const void *identity = nullptr;                        // what the data belongs to; a change regenerates
    uint64_t stamp = 0;                                    // changes whenever the data does
};

// Three axis-aligned slices of an adaptive level-set grid, drawn into ImGui: the plane at a fixed X
// (Y right, Z up), at a fixed Y (X right, Z up) and at a fixed Z (X right, Y up), each with a
// slider below it to move the slice. Each pixel is colored by the octree depth of the leaf it
// falls in (depth 0 for uniform grids), block outlines are darker, the inside of the surface is
// lighter, and the surface's exact cross-section is drawn in white. Needs a current OpenGL 3 context (the slices are textures).
class TriPlanarView
{
public:
    static constexpr int kResolution = 256;  // pixels per slice side

    ~TriPlanarView();
    // Draws the three slices of `source`, sized to fit the remaining space of the current ImGui window
    // without scrolling.
    void draw(const TriPlanarSource &source);

private:
    struct Plane {
        unsigned int texture = 0;
        float slice = -1.0f;  // voxels along the plane's normal axis (< 0: not set yet)
        // What the texture shows, to regenerate only on change.
        uint64_t shownStamp = 0;
        float shownSlice = -1.0f;
        const void *shownIdentity = nullptr;
    };
    std::array<Plane, 3> mPlanes;
    std::vector<unsigned char> mPixels;

    void render(const TriPlanarSource &source, int axis);
};
}  // namespace springls_gui
