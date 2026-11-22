#pragma once
#include <core/Common3D.hpp>

#include <Eigen/Geometry>

namespace sphere_gui
{
// An orbit camera about a target point with a free orientation (no fixed up axis), so it can follow an object
// over a pole of a closed surface without flipping. Controls as the mesh window's orbit camera: left drag
// orbits (about the view's own axes), Shift + left drag or middle drag pans, right drag and the wheel zoom.
// View and projection follow OpenGL conventions (view space looks down -z, NDC depth -1..1).
struct Camera {
    Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();  // camera to world: the eye is on its +z
    springls::Vec3 target = springls::Vec3::Zero();
    float distance = 1.0f, fovY = 0.6f;
    float radius = 1.0f;  // scene radius (set by fit()), bounds the zoom range

    // Frames the box [lo, hi] from the mesh window's default direction (y up).
    void fit(const springls::Vec3 &lo, const springls::Vec3 &hi);
    // Looks along -direction (the eye on the +direction side of the target), keeping the view's up as close to
    // the current one as possible.
    void lookFrom(const springls::Vec3 &direction);
    // Turns the camera (its orientation and its offset from the target) by `rotation`.
    void turn(const Eigen::Quaternionf &rotation) { orientation = (rotation * orientation).normalized(); }
    // Applies the mouse state of the current ImGui frame; a no-op while ImGui wants the mouse.
    void handleInput();
    springls::Vec3 eye() const { return target + orientation * springls::Vec3(0.0f, 0.0f, distance); }
    Eigen::Matrix4f view() const;
    // Near = distance * 0.01, far = distance * 100.
    Eigen::Matrix4f projection(float aspect) const;
};
}  // namespace sphere_gui
