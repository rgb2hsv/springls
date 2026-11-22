#include "Offscreen.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include <stb_image_write.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <vector>

namespace springls_gui
{
OffscreenTarget::~OffscreenTarget()
{
    if (mFramebuffer) glDeleteFramebuffers(1, &mFramebuffer);
    if (mRenderbuffers[0]) glDeleteRenderbuffers(2, mRenderbuffers);
}

bool OffscreenTarget::init(int width, int height)
{
    mWidth = width;
    mHeight = height;
    glGenFramebuffers(1, &mFramebuffer);
    glGenRenderbuffers(2, mRenderbuffers);
    glBindRenderbuffer(GL_RENDERBUFFER, mRenderbuffers[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, mRenderbuffers[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mRenderbuffers[0]);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, mRenderbuffers[1]);
    const bool complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return complete;
}

void OffscreenTarget::bind() const { glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer); }

bool OffscreenTarget::writePng(const std::filesystem::path &path) const
{
    std::vector<unsigned char> pixels(static_cast<size_t>(mWidth) * mHeight * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebuffer);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    for (size_t i = 3; i < pixels.size(); i += 4) pixels[i] = 255;  // opaque, whatever blending left in alpha
    stbi_flip_vertically_on_write(1);
    return stbi_write_png(path.string().c_str(), mWidth, mHeight, 4, pixels.data(), mWidth * 4) != 0;
}
}  // namespace springls_gui
