// The one NanoVG OpenGL 3 backend implementation of the windows (the 2D and 3D Enright renderers use it).
#include "GL.hpp"

#include "nanovg.hpp"
#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg_gl.hpp"

// nanovg_gl.hpp defines deleteGL inline in its implementation section, which the renderers do not
// include: taking its address here emits the copy they link against.
extern void (*const kNanovgDeleteGL)(std::unique_ptr<nvg::Context> &);
void (*const kNanovgDeleteGL)(std::unique_ptr<nvg::Context> &) = &nvg::deleteGL;
