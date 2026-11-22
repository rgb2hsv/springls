#pragma once
#include "../mesh/Matcap.hpp"

#include <core/Common3D.hpp>
#include <core/SphereSimulation.hpp>

#include <string>
#include <vector>

namespace sphere_gui
{
struct Layers {
    bool regions = true, isolines = true, springls = true, particles = true, vertices = true, normals = true, contour = true, arrows = true, reference = true,
         wireframe = true;
    float isoSpacing = 0.5f;  // isoline spacing, in length scales
    int matcap = 0;           // index into the loaded matcaps
};

// Draws a SphereSnapshot in an OpenGL 3.3 core context: the input mesh with matcap shading, tinted by region
// with a sharp boundary at the band's linear zero crossing (per fragment; the background untinted), with
// (colors from the toybox palette) the triangulation's wireframe colored by subdivision level; isolines of the band's distance every isoSpacing length scales out to the band's width; the springls
// (gray, v0 -> particle -> v1, round caps), their vertexes (yellow), normals (green) and particles (dark red), sized on the
// surface relative to the sphere, the contour (teal, 4 px) as
// in the 2D window, the Lagrangian reference (thin yellow) and the velocity arrows (fuchsia, with heads) as
// screen-space lines and dots pulled slightly toward the eye so they sit on the surface.
class Renderer
{
public:
    Renderer() = default;
    Renderer(const Renderer &) = delete;
    Renderer &operator=(const Renderer &) = delete;
    // Releases GL objects: destroy while the GL context is still current.
    ~Renderer();
    // Compiles the shaders and uploads the matcaps. Needs a current GL 3.3 context.
    bool init(const std::vector<viewer::Matcap> &matcaps, std::string &error);
    // Uploads the snapshot's buffers; draw() shows the last one uploaded.
    void setSnapshot(const toybox::SphereSnapshot &snapshot);
    bool hasSnapshot() const { return mUploaded; }
    // Draws into the window rectangle (x, y, width, height), in framebuffer pixels with y up, with OpenGL-style
    // view and projection matrices. Clears only the depth buffer in that rectangle; the caller clears the color.
    void draw(const Layers &layers, const Eigen::Matrix4f &view, const Eigen::Matrix4f &projection, int x, int y, int width, int height);

private:
    struct Buffer {
        unsigned int vao = 0, vbo = 0;
        int count = 0;  // vertexes
    };
    void release();

    unsigned int mSurfaceProgram = 0, mLineProgram = 0, mCapsuleProgram = 0, mFlatProgram = 0;
    std::vector<unsigned int> mMatcaps;
    unsigned int mFallback = 0;                                 // 1x1 white
    int mBackgroundMatcap = -1;                                 // index into mMatcaps of the background's (kBackgroundMatcap), -1: none
    Buffer mSurface, mSpringls, mContour, mReference, mArrows;  // mSpringls: springls, vertexes, particles (capsules)
    float mSceneRadius = 1.0f;                                  // the sphere's radius (springl, particle and vertex sizes are relative to it)
    size_t mRadiusVertices = 0;
    float mLengthScale = 0, mBandWidth = 0;
    bool mUploaded = false;
    // Whether the input mesh is closed (every edge shared by two triangles), cached per mesh size: a
    // reference polyline is closed on a closed mesh.
    size_t mClosedVertices = 0, mClosedTriangles = 0;
    bool mClosed = false;
};
}  // namespace sphere_gui
