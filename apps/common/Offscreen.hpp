#pragma once
#include "GL.hpp"

#include <filesystem>

namespace springls_gui
{
// Offscreen color + depth/stencil target for screenshots and frames: a hidden window's default framebuffer
// may not hold its pixels. Needs a current OpenGL 3.3 context; destroy it while the context is current.
class OffscreenTarget
{
public:
    OffscreenTarget() = default;
    OffscreenTarget(const OffscreenTarget &) = delete;
    OffscreenTarget &operator=(const OffscreenTarget &) = delete;
    ~OffscreenTarget();
    // False when the framebuffer is incomplete.
    bool init(int width, int height);
    void bind() const;
    // Reads the pixels back (opaque) and writes them, top row first, as PNG.
    bool writePng(const std::filesystem::path &path) const;
    int width() const { return mWidth; }
    int height() const { return mHeight; }

private:
    GLuint mFramebuffer = 0;
    GLuint mRenderbuffers[2] = {0, 0};
    int mWidth = 0, mHeight = 0;
};
}  // namespace springls_gui
