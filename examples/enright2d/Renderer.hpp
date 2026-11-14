#pragma once
#include "Experiment.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nvg
{
class Context;
}

namespace enright
{
// What the panels show.
struct Visibility {
    bool springls = true;
    bool capsules = false;
    bool particles = true;
    bool vertexes = true;
    bool normals = true;
    bool contour = true;
    bool contourFill = true;
    bool referenceCircle = true;
    bool grid = true;  // level-set grid (adaptive blocks colored by depth)
    bool primal = true;
    bool dual = true;
};

// Shared region of interest in domain units: the square [origin, origin + size]^2.
struct RegionOfInterest {
    springls::Vec2 origin = springls::Vec2(0.0f, 0.0f);
    float size = 1.0f;
};

struct Rect {
    float x = 0, y = 0, w = 0, h = 0;
    bool contains(float px, float py) const { return px >= x && py >= y && px <= x + w && py <= y + h; }
};

// Draws one panel per run (2 columns when there is more than one) with NanoVG, each with a
// text header of its metrics. All panels share one region of interest.
class Renderer
{
public:
    static constexpr float kPanelMargin = 30.0f;
    static constexpr float kHeaderHeight = 175.0f;

    Renderer();
    ~Renderer();
    // Creates the NanoVG GL3 context; needs a current OpenGL 3 context.
    bool init(const std::string &fontFile, std::string &error);
    void shutdown();

    // Panel rectangles for `count` panels in `bounds`.
    static std::vector<Rect> Layout(size_t count, const Rect &bounds);
    // Size of an image that fits `count` panels of panelSize pixels (rounded to even numbers).
    static void ImageSize(size_t count, float panelSize, int &width, int &height);

    // Draws the experiment into `bounds` (logical pixels) of a framebuffer whose logical size is
    // viewWidth x viewHeight. `status` is appended to the time line of unfinished runs.
    void draw(const Experiment &experiment, const Visibility &visibility, const RegionOfInterest &roi, const Rect &bounds, float viewWidth, float viewHeight,
              float devicePixelRatio, const std::string &status);
    const std::vector<Rect> &panels() const { return mPanels; }

private:
    std::unique_ptr<nvg::Context> mContext;
    int mFont = -1;
    std::vector<Rect> mPanels;

    void drawRun(const Experiment &experiment, const Run &run, const Visibility &visibility, const RegionOfInterest &roi, const Rect &panel,
                 const std::string &status, float headerScale);
    void drawGrid(const Experiment &experiment, const Run &run, const RegionOfInterest &roi, const Rect &panel);
};

// Renders the panels (no GUI widgets) into an offscreen framebuffer and saves numbered PNGs:
// <directory>/frame_000000.png, frame_000001.png, ...
// Existing frames are never overwritten: if the requested directory already has files, init()
// records into the first free sibling <directory>_001, <directory>_002, ... instead.
class FrameRecorder
{
public:
    // `requested` if it does not exist or is empty, else the first free requested_NNN.
    static std::filesystem::path UnusedDirectory(const std::filesystem::path &requested);
    ~FrameRecorder();
    bool init(const std::filesystem::path &directory, int width, int height, std::string &error);
    bool active() const { return mFramebuffer != 0; }
    // Draws and writes the next frame; returns false on a write error.
    bool capture(Renderer &renderer, const Experiment &experiment, const Visibility &visibility, const RegionOfInterest &roi);
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
}  // namespace enright
