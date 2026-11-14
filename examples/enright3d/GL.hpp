#pragma once
// OpenGL 3 + GLFW headers. Linux/macOS use the system GL prototypes (GL_GLEXT_PROTOTYPES);
// Windows uses GLEW (SPRINGLS_GLEW).
#ifdef SPRINGLS_GLEW
#include <GL/glew.h>
#endif
#ifdef __APPLE__
#define GLFW_INCLUDE_GLCOREARB
#else
#define GLFW_INCLUDE_GLEXT
#endif
#include <GLFW/glfw3.h>
