#include "Camera.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace sphere_gui
{
using springls::Vec3;
namespace
{
constexpr float kDefaultYaw = 0.6f, kDefaultPitch = 0.3f;  // the mesh window's orbit camera's
}  // namespace

void Camera::fit(const Vec3 &lo, const Vec3 &hi)
{
    target = 0.5f * (lo + hi);
    radius = std::max(0.5f * (hi - lo).norm(), 1e-6f);
    distance = radius / std::sin(fovY * 0.5f) * 1.05f;
    // Yaw about +y, then pitch up about the camera's x axis.
    orientation = Eigen::AngleAxisf(kDefaultYaw, Vec3::UnitY()) * Eigen::AngleAxisf(-kDefaultPitch, Vec3::UnitX());
}

void Camera::lookFrom(const Vec3 &direction)
{
    const float norm = direction.norm();
    if (!(norm > 0.0f)) return;
    const Vec3 back = direction / norm;  // camera +z
    Vec3 up = orientation * Vec3::UnitY();
    up -= back * back.dot(up);
    if (up.norm() < 1e-4f) up = (orientation * Vec3::UnitZ()).cross(back);  // looking along the old up: any perpendicular
    if (up.norm() < 1e-4f) up = back.unitOrthogonal();
    up.normalize();
    Eigen::Matrix3f rotation;
    rotation.col(0) = up.cross(back);
    rotation.col(1) = up;
    rotation.col(2) = back;
    orientation = Eigen::Quaternionf(rotation).normalized();
}

Eigen::Matrix4f Camera::view() const
{
    const Eigen::Matrix3f rotation = orientation.toRotationMatrix().transpose();
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = rotation;
    m.block<3, 1>(0, 3) = -(rotation * eye());
    return m;
}

Eigen::Matrix4f Camera::projection(float aspect) const
{
    const float nearPlane = distance * 0.01f, farPlane = distance * 100.0f;
    const float f = 1.0f / std::tan(fovY * 0.5f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = f / aspect;
    m(1, 1) = f;
    m(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
    m(2, 3) = 2.0f * farPlane * nearPlane / (nearPlane - farPlane);
    m(3, 2) = -1.0f;
    return m;
}

void Camera::handleInput()
{
    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;
    const float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
    const Vec3 right = orientation * Vec3::UnitX(), up = orientation * Vec3::UnitY();
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.KeyShift && (dx != 0.0f || dy != 0.0f)) {
        // Dragging right turns the eye left about the view's up; dragging down raises it about the view's right.
        turn(Eigen::Quaternionf(Eigen::AngleAxisf(-dx * 0.005f, up)) * Eigen::Quaternionf(Eigen::AngleAxisf(-dy * 0.005f, right)));
    }
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) || (ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyShift)) {
        const float scale = distance * 0.0015f;
        target = target - right * (dx * scale) + up * (dy * scale);
    }
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) distance *= std::exp(dy * 0.005f);
    if (io.MouseWheel != 0.0f) distance *= std::exp(-io.MouseWheel * 0.1f);
    distance = std::clamp(distance, radius * 0.001f, radius * 100.0f);
}
}  // namespace sphere_gui
