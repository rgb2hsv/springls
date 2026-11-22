#pragma once
#include <core/Enright3DSimulation.hpp>
#include <core/Experiment.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nvg
{
class Context;
}

namespace enright3d
{
using toybox::Experiment;
using toybox::ExperimentSnapshot;
using toybox::Pacing;
using toybox::Run;
using toybox::RunSnapshot;

// What the panels show. Layers stack iso-surface, dual springls, primal springls (front).
struct Visibility {
    bool isoSurface = true;
    bool primal = true;  // all springls for single-constellation springls
    bool dual = true;
    bool particles = false;  // hidden by default; the Particles toggle shows them
    // Springls are drawn shrunk about their particles by this factor so single elements stand out.
    float springlScale = 0.85f;
    // Capsules: each springl element grown by capsuleRadius in its own plane (the region it covers
    // in the fill gap test, by default the fill distance), drawn as a solid region with the element
    // itself (the inlay) in the constellation color and the overshoot in one capsule color.
    bool capsules = false;
    float capsuleRadius = springls::SpringLevelSet3D::Tuning().fillDistance;
};

// Orthographic orbit camera shared by all panels. The domain is mapped to a unit cube about the
// origin; yaw / pitch rotate it, pan moves it in the view plane, and zoom scales the view.
//
// The default pose suits the Enright test (from the exact flow, the initial sphere advected with
// RK4): it looks at the flat disk the sphere becomes at t = T/2 from the side the disk moves
// toward as it flattens, tilted 35 degrees from the disk normal toward the axis the sheet tumbles
// about (which lies in the disk's plane), so the face stays visible (cos 35 = 0.82), motion comes
// toward the camera, and the swirl reads counterclockwise. Pan and zoom fit the shape at all times.
struct OrbitCamera {
    float yaw = -0.1754f;    // radians, about the vertical axis
    float pitch = -0.1614f;  // radians, about the horizontal axis
    float zoom = 2.05f;
    float panX = -0.0210f, panY = 0.0213f;  // view-plane offset, in unit-cube units
    void rotate(float dx, float dy);        // by screen pixels / 200
    void reset() { *this = OrbitCamera(); }
};

struct Rect {
    float x = 0, y = 0, w = 0, h = 0;
    bool contains(float px, float py) const { return px >= x && py >= y && px <= x + w && py <= y + h; }
};

// Draws one panel per run (2 columns when there is more than one) with OpenGL 3, each with a text
// header of its metrics (NanoVG). All panels share one camera.
class Renderer
{
public:
    static constexpr float kPanelMargin = 30.0f;
    static constexpr float kHeaderHeight = 175.0f;

    Renderer();
    ~Renderer();
    // Compiles the shaders and creates the NanoVG context; needs a current OpenGL 3.3 context.
    bool init(const std::string &fontFile, std::string &error);
    void shutdown();

    static std::vector<Rect> Layout(size_t count, const Rect &bounds);
    // Size of an image that fits `count` panels of panelSize pixels (rounded to even numbers).
    static void ImageSize(size_t count, float panelSize, int &width, int &height);

    // Draws the experiment into `bounds` (logical pixels) of the bound framebuffer, whose logical
    // size is viewWidth x viewHeight. `status` is appended to the time line of unfinished runs.
    void draw(const ExperimentSnapshot &experiment, const Visibility &visibility, const OrbitCamera &camera, const Rect &bounds, float viewWidth,
              float viewHeight, float devicePixelRatio, const std::string &status);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

// Renders the panels (no GUI widgets) into an offscreen framebuffer and saves numbered PNGs:
// <directory>/frame_000000.png, frame_000001.png, ... Existing frames are never overwritten: if
// the requested directory already has files, init() records into the first free sibling
// <directory>_001, <directory>_002, ... instead.
class FrameRecorder
{
public:
    static std::filesystem::path UnusedDirectory(const std::filesystem::path &requested);
    ~FrameRecorder();
    bool init(const std::filesystem::path &directory, int width, int height, std::string &error);
    bool active() const { return mFramebuffer != 0; }
    // Draws and writes the next frame; returns false on a write error.
    bool capture(Renderer &renderer, const ExperimentSnapshot &experiment, const Visibility &visibility, const OrbitCamera &camera);
    uint64_t frameCount() const { return mFrame; }
    const std::filesystem::path &directory() const { return mDirectory; }

private:
    std::filesystem::path mDirectory;
    int mWidth = 0, mHeight = 0;
    unsigned int mFramebuffer = 0, mColor = 0, mDepthStencil = 0;
    uint64_t mFrame = 0;
    std::vector<unsigned char> mPixels;
    void release();
};
}  // namespace enright3d
