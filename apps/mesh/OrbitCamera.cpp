#include "OrbitCamera.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace viewer
{
using springls::Vec3;
namespace
{
constexpr float kMaxPitch = 1.55f;  // keeps the view off the poles
}  // namespace

void OrbitCamera::fit(const Vec3 &lo, const Vec3 &hi)
{
    target = 0.5f * (lo + hi);
    radius = std::max(0.5f * (hi - lo).norm(), 1e-6f);
    distance = radius / std::sin(fovY * 0.5f) * 1.05f;
}

Vec3 OrbitCamera::eye() const
{
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const Vec3 direction = zUp ? Vec3(cp * std::cos(yaw), cp * std::sin(yaw), sp) : Vec3(cp * std::sin(yaw), sp, cp * std::cos(yaw));
    return target + direction * distance;
}

Eigen::Matrix4f OrbitCamera::view() const
{
    const Vec3 position = eye();
    const Vec3 forward = (target - position).normalized();
    const Vec3 right = forward.cross(up()).normalized();
    const Vec3 cameraUp = right.cross(forward);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<1, 3>(0, 0) = right.transpose();
    m.block<1, 3>(1, 0) = cameraUp.transpose();
    m.block<1, 3>(2, 0) = -forward.transpose();
    m(0, 3) = -right.dot(position);
    m(1, 3) = -cameraUp.dot(position);
    m(2, 3) = forward.dot(position);
    return m;
}

Eigen::Matrix4f OrbitCamera::projection(float aspect) const
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

void OrbitCamera::handleInput()
{
    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }
    const float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.KeyShift) {
        yaw -= dx * 0.005f;
        pitch = std::clamp(pitch + dy * 0.005f, -kMaxPitch, kMaxPitch);
    }
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) || (ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyShift)) {
        const Vec3 forward = (target - eye()).normalized();
        const Vec3 right = forward.cross(up()).normalized();
        const Vec3 cameraUp = right.cross(forward);
        const float scale = distance * 0.0015f;
        target = target - right * (dx * scale) + cameraUp * (dy * scale);
    }
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        distance *= std::exp(dy * 0.005f);
    }
    if (io.MouseWheel != 0.0f) {
        distance *= std::exp(-io.MouseWheel * 0.1f);
    }
    distance = std::clamp(distance, radius * 0.001f, radius * 100.0f);
}
}  // namespace viewer
