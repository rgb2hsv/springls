#pragma once
#include <core/Common3D.hpp>

namespace viewer
{
// Orbit camera about a target point (meshlets' controls): left drag orbits, Shift + left drag or middle
// drag pans, right drag and the wheel zoom. View and projection follow OpenGL conventions (right-handed
// view space looking down -z, y-up clip space, NDC depth -1..1).
struct OrbitCamera {
    float yaw = 0.6f, pitch = 0.3f, distance = 1.0f, fovY = 0.6f;
    springls::Vec3 target = springls::Vec3::Zero();
    bool zUp = false;
    float radius = 1.0f;  // scene radius (set by fit()), bounds the zoom range

    // Frames the box [lo, hi]: orbit about its center from far enough that its bounding sphere fits.
    void fit(const springls::Vec3 &lo, const springls::Vec3 &hi);
    // Applies the mouse state of the current ImGui frame; a no-op while ImGui wants the mouse.
    void handleInput();
    Eigen::Matrix4f view() const;
    // Near = distance * 0.01, far = distance * 100.
    Eigen::Matrix4f projection(float aspect) const;
    springls::Vec3 eye() const;
    springls::Vec3 up() const { return zUp ? springls::Vec3::UnitZ() : springls::Vec3::UnitY(); }
};
}  // namespace viewer
