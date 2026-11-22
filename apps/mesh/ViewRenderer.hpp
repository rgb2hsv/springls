#pragma once
#include "../common/Palette.hpp"
#include "Matcap.hpp"
#include "MeshTypes.hpp"
#include "OrbitCamera.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace viewer
{
struct Layers {
    bool isoSurface = true, primal = true, dual = true, particles = false, inputMesh = false;
    float springlScale = 0.85f;
    float particleSize = 0.25f;                                                // particle diameter in finest cells, so it scales with the scene
    std::array<float, 3> primalColor = toybox::palette::kPrimalSpringl.rgb();  // the toybox palette's (as every window)
    std::array<float, 3> dualColor = toybox::palette::kDualSpringl.rgb();
    int matcap = 0;  // index into the loaded matcaps
};

// Draws a SessionSnapshot with matcap shading in an OpenGL 3.3 core context. The iso-surface, primal
// springls and dual springls are coplanar, so they are stacked with polygon offsets: the primal layer
// wins over the dual layer, which wins over the iso-surface.
class ViewRenderer
{
public:
    ViewRenderer() = default;
    ViewRenderer(const ViewRenderer &) = delete;
    ViewRenderer &operator=(const ViewRenderer &) = delete;
    // Releases GL objects: destroy while the GL context is still current.
    ~ViewRenderer();
    // Compiles the shaders and uploads the matcaps. Needs a current GL 3.3 context (and loaded entry points).
    bool init(const std::vector<Matcap> &matcaps, std::string &error);
    // GL texture of a matcap (for ImGui thumbnails); 0 when out of range.
    unsigned int matcapTexture(int index) const;
    // Draws into the window rectangle (x, y, width, height), in framebuffer pixels with y up. Clears only
    // the depth buffer in that rectangle; the caller clears the color. Buffers are rebuilt when
    // snapshot.stamp changes.
    void draw(const SessionSnapshot &snapshot, const Layers &layers, const OrbitCamera &camera, int x, int y, int width, int height);

private:
    struct Mesh {
        unsigned int vao = 0, vbo = 0, ebo = 0;
        int indexCount = 0;
    };
    struct Points {
        unsigned int vao = 0, vbo = 0;
        int count = 0;
    };
    void upload(const SessionSnapshot &snapshot);
    void release();

    unsigned int mSurfaceProgram = 0, mPointProgram = 0;
    std::vector<unsigned int> mMatcaps;
    unsigned int mFallback = 0;  // 1x1 white
    Mesh mIso, mInput, mPrimal, mDual;
    Points mPrimalParticles, mDualParticles;
    uint64_t mStamp = 0;
    bool mUploaded = false;
};
}  // namespace viewer
