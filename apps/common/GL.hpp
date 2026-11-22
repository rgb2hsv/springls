#pragma once
// OpenGL 3 + GLFW headers. Linux/macOS use the system GL prototypes (GL_GLEXT_PROTOTYPES);
// Windows loads the entry points with the glad header bundled in GLFW's deps (SPRINGLS_GLAD).
#ifdef SPRINGLS_GLAD
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#elif defined(__APPLE__)
#define GLFW_INCLUDE_GLCOREARB
#else
#define GLFW_INCLUDE_GLEXT
#endif
#include <GLFW/glfw3.h>

namespace toybox
{

// Loads the OpenGL entry points for the current context (a no-op without a loader). False when that fails.
inline bool LoadGL()
{
#ifdef SPRINGLS_GLAD
    return gladLoadGL(glfwGetProcAddress) != 0;
#else
    return true;
#endif
}

}  // namespace toybox
