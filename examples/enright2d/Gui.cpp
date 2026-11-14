#include "Gui.hpp"

#include "GL.hpp"
#include "Renderer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <iostream>

namespace enright
{
using namespace springls;
namespace
{
constexpr float kSidebarWidth = 300.0f;
constexpr float kZoomPerScrollTick = 0.1f;
// ROI size limits in domain units.
constexpr float kMinRoiSize = 0.01f;
constexpr float kMaxRoiSize = 4.0f;
constexpr int kMaxStepsPerFrame = 50;

void GlfwError(int code, const char *description) { std::cerr << "GLFW error " << code << ": " << description << std::endl; }

// Creates an OpenGL 3.3 core window (hidden when !visible) and makes it current.
GLFWwindow *CreateWindow(int width, int height, bool visible, const std::string &title, std::string &error)
{
    glfwSetErrorCallback(GlfwError);
    if (!glfwInit()) {
        error = "could not initialize GLFW (is a display available?)";
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_SAMPLES, visible ? 4 : 0);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        error = "could not create an OpenGL 3.3 window";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
#ifdef SPRINGLS_GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        error = "could not initialize GLEW";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }
    glGetError();  // GLEW may leave GL_INVALID_ENUM behind
#endif
    return window;
}

void DestroyWindow(GLFWwindow *window)
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool StartRecorder(FrameRecorder &recorder, const Experiment &experiment, const DisplayOptions &options)
{
    if (options.framesDir.empty()) return true;
    int width = 0, height = 0;
    Renderer::ImageSize(experiment.runs().size(), options.framePanelSize, width, height);
    std::string error;
    if (!recorder.init(options.framesDir, width, height, error)) {
        std::cerr << "error: " << error << std::endl;
        return false;
    }
    std::cout << "Saving " << width << " x " << height << " frames to " << recorder.directory().string() << std::endl;
    return true;
}

bool Capture(FrameRecorder &recorder, Renderer &renderer, const Experiment &experiment, const Visibility &visibility)
{
    if (!recorder.active()) return true;
    // Saved frames always show the whole domain.
    if (!recorder.capture(renderer, experiment, visibility, RegionOfInterest())) {
        std::cerr << "error: could not write frame " << recorder.frameCount() << " to " << recorder.directory().string() << std::endl;
        return false;
    }
    return true;
}
}  // namespace

int RunOffscreenFrames(Experiment &experiment, const DisplayOptions &options)
{
    std::string error;
    GLFWwindow *window = CreateWindow(64, 64, false, options.windowTitle, error);
    if (!window) {
        std::cerr << "error: " << error << std::endl;
        return 1;
    }
    int status = 0;
    {
        Renderer renderer;
        FrameRecorder recorder;
        const Visibility visibility;
        if (!renderer.init(options.fontFile, error)) {
            std::cerr << "error: " << error << std::endl;
            status = 1;
        } else if (!StartRecorder(recorder, experiment, options) || !Capture(recorder, renderer, experiment, visibility)) {
            status = 1;
        } else {
            while (experiment.step()) {
                if (!Capture(recorder, renderer, experiment, visibility)) {
                    status = 1;
                    break;
                }
            }
            experiment.printSummary(std::cout);
            std::cout << "Wrote " << recorder.frameCount() << " frames to " << recorder.directory().string() << std::endl;
        }
    }
    DestroyWindow(window);
    return status;
}

int RunGui(Experiment &experiment, const DisplayOptions &options)
{
    std::string error;
    // Fit the window to the monitor work area.
    int width = 1700, height = 1100;
    if (glfwInit()) {
        int x, y, w, h;
        if (GLFWmonitor *monitor = glfwGetPrimaryMonitor()) {
            glfwGetMonitorWorkarea(monitor, &x, &y, &w, &h);
            if (w > 0 && h > 0) {
                width = std::min(width, static_cast<int>(0.95f * w));
                height = std::min(height, static_cast<int>(0.9f * h));
            }
        }
    }
    GLFWwindow *window = CreateWindow(width, height, true, options.windowTitle, error);
    if (!window) {
        std::cerr << "error: " << error << " (use --no-gui to run without a window)" << std::endl;
        return 1;
    }
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    int status = 0;
    {
        Renderer renderer;
        FrameRecorder recorder;
        Visibility visibility;
        RegionOfInterest roi;
        bool paused = options.startPaused;
        bool summaryPrinted = false;
        bool dragging = false;
        int stepsPerFrame = std::clamp(options.stepsPerFrame, 1, kMaxStepsPerFrame);
        ImVec2 lastDrag;
        bool recording = false;
        if (!renderer.init(options.fontFile, error)) {
            std::cerr << "error: " << error << std::endl;
            status = 1;
        } else {
            recording = StartRecorder(recorder, experiment, options) && Capture(recorder, renderer, experiment, visibility);
            if (!options.framesDir.empty() && !recording) status = 1;
        }
        const auto restart = [&]() {
            std::string message;
            if (!experiment.init(experiment.options(), message)) std::cerr << "error: " << message << std::endl;
            summaryPrinted = false;
            if (recording) recording = StartRecorder(recorder, experiment, options) && Capture(recorder, renderer, experiment, visibility);
        };
        while (status == 0 && !glfwWindowShouldClose(window)) {
            glfwPollEvents();
            if (!paused) {
                for (int k = 0; k < stepsPerFrame; k++) {
                    if (!experiment.step()) break;
                    if (recording && !Capture(recorder, renderer, experiment, visibility)) recording = false;
                }
            }
            if (!summaryPrinted && experiment.done()) {
                experiment.printSummary(std::cout);
                summaryPrinted = true;
                if (options.exitWhenDone) break;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            int winW, winH, fbW, fbH;
            glfwGetWindowSize(window, &winW, &winH);
            glfwGetFramebufferSize(window, &fbW, &fbH);
            const float pixelRatio = (winW > 0) ? static_cast<float>(fbW) / winW : 1.0f;

            // Sidebar.
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(kSidebarWidth, static_cast<float>(winH)));
            ImGui::Begin("Enright 2D", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            if (ImGui::Button(paused ? "Play" : "Pause", ImVec2(80, 0))) paused = !paused;
            ImGui::SameLine();
            if (ImGui::Button("Step", ImVec2(80, 0)) && experiment.step()) {
                if (recording && !Capture(recorder, renderer, experiment, visibility)) recording = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Restart", ImVec2(80, 0))) restart();
            ImGui::SliderInt("steps / frame", &stepsPerFrame, 1, kMaxStepsPerFrame);
            if (ImGui::Button("Reset view")) roi = RegionOfInterest();
            if (experiment.runs().size() > 1) {
                int pacing = static_cast<int>(experiment.pacing());
                const char *pacings[] = {"Lockstep", "Equal compute", "Same sim time"};
                if (ImGui::Combo("pacing", &pacing, pacings, IM_ARRAYSIZE(pacings))) experiment.setPacing(static_cast<Pacing>(pacing));
                ImGui::TextDisabled("M: cycle pacing");
            }
            ImGui::TextDisabled("Space: pause  R: reset view\nDrag: pan  Scroll: zoom");
            if (recorder.active()) {
                ImGui::Separator();
                ImGui::Text("Recording %s", recording ? "on" : "stopped");
                ImGui::TextWrapped("%llu frames -> %s", static_cast<unsigned long long>(recorder.frameCount()), recorder.directory().string().c_str());
            }
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Elements", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Springls", &visibility.springls);
                ImGui::Checkbox("Capsules", &visibility.capsules);
                ImGui::Checkbox("Particles", &visibility.particles);
                ImGui::Checkbox("Vertexes", &visibility.vertexes);
                ImGui::Checkbox("Normals", &visibility.normals);
            }
            if (ImGui::CollapsingHeader("Constellations", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Primal", &visibility.primal);
                ImGui::Checkbox("Dual", &visibility.dual);
            }
            if (ImGui::CollapsingHeader("Interface", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Contour", &visibility.contour);
                ImGui::Checkbox("Contour fill", &visibility.contourFill);
                ImGui::Checkbox("Reference circle", &visibility.referenceCircle);
                ImGui::Checkbox("Grid", &visibility.grid);
            }
            if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (const Run &run : experiment.runs()) {
                    const double perStep = run.sim->getIteration() > 0 ? 1000.0 * run.seconds / run.sim->getIteration() : 0.0;
                    ImGui::Text("%-28s %6.1f ms/step", MethodTitle(run.method), perStep);
                    ImGui::Text("  %llu sub-steps, %.1f s", static_cast<unsigned long long>(run.sim->getAdvectionSubstepCount()), run.seconds);
                }
                ImGui::Text("threads: %d", MaxThreads());
            }
            ImGui::End();

            // Pan / zoom on the panels.
            const Rect view{kSidebarWidth, 0.0f, std::max(0.0f, winW - kSidebarWidth), static_cast<float>(winH)};
            const std::vector<Rect> panels = Renderer::Layout(experiment.runs().size(), view);
            const ImVec2 cursor = io.MousePos;
            int hovered = -1;
            for (size_t i = 0; i < panels.size(); i++) {
                if (panels[i].contains(cursor.x, cursor.y)) hovered = static_cast<int>(i);
            }
            if (!io.WantCaptureMouse) {
                if (hovered >= 0 && io.MouseWheel != 0.0f) {
                    // Zoom about the cursor: the domain point under it stays fixed.
                    const Rect &p = panels[hovered];
                    const Vec2 rel((cursor.x - p.x) / p.w, 1.0f - (cursor.y - p.y) / p.h);
                    const Vec2 anchor = roi.origin + rel * roi.size;
                    const float size = std::clamp(roi.size * (1.0f - kZoomPerScrollTick * io.MouseWheel), kMinRoiSize, kMaxRoiSize);
                    roi.origin = anchor - rel * size;
                    roi.size = size;
                }
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && hovered >= 0) {
                    dragging = true;
                    lastDrag = cursor;
                }
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) dragging = false;
            if (dragging && !panels.empty() && panels[0].w > 0.0f) {
                const float scale = roi.size / panels[0].w;
                roi.origin -= Vec2((cursor.x - lastDrag.x) * scale, -(cursor.y - lastDrag.y) * scale);
                lastDrag = cursor;
            }
            if (!io.WantCaptureKeyboard) {
                if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) paused = !paused;
                if (ImGui::IsKeyPressed(ImGuiKey_M, false) && experiment.runs().size() > 1) {
                    experiment.setPacing(static_cast<Pacing>((static_cast<int>(experiment.pacing()) + 1) % 3));
                }
                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) roi = RegionOfInterest();
            }

            ImGui::Render();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbW, fbH);
            glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            renderer.draw(experiment, visibility, roi, view, static_cast<float>(winW), static_cast<float>(winH), pixelRatio,
                          paused ? "  (paused - Space to resume)" : "");
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    DestroyWindow(window);
    return status;
}
}  // namespace enright
