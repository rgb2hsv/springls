#include "Gui.hpp"

#include "GL.hpp"
#include "Renderer.hpp"
#include "TriPlanar.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <thread>

namespace enright3d
{
using namespace springls;
namespace
{
constexpr float kSidebarWidth = 300.0f;
constexpr float kSliceSidebarWidth = 300.0f;  // right sidebar: tri-planar slices of the grid
constexpr float kZoomPerScrollTick = 0.1f;
constexpr float kMinZoom = 0.2f, kMaxZoom = 40.0f;

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
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
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
    if (!toybox::LoadGL()) {
        error = "could not load the OpenGL 3.3 entry points";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }
    return window;
}

void DestroyWindow(GLFWwindow *window)
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool StartRecorder(FrameRecorder &recorder, size_t runCount, const DisplayOptions &options)
{
    if (options.framesDir.empty()) return true;
    int width = 0, height = 0;
    Renderer::ImageSize(runCount, options.framePanelSize, width, height);
    std::string error;
    if (!recorder.init(options.framesDir, width, height, error)) {
        std::cerr << "error: " << error << std::endl;
        return false;
    }
    std::cout << "Saving " << width << " x " << height << " frames to " << recorder.directory().string() << std::endl;
    return true;
}

bool Capture(FrameRecorder &recorder, Renderer &renderer, const ExperimentSnapshot &snapshot, const Visibility &visibility, const OrbitCamera &camera)
{
    if (!recorder.active()) return true;
    if (!recorder.capture(renderer, snapshot, visibility, camera)) {
        std::cerr << "error: could not write frame " << recorder.frameCount() << " to " << recorder.directory().string() << std::endl;
        return false;
    }
    return true;
}

// Steps the experiment on a worker thread, so a slow 3D step never stalls the UI (ImGui spreads
// queued input over frames, so at a few frames per second clicks and scrolling lag badly). After
// each step the worker publishes a snapshot; the UI thread draws the latest one and touches the
// experiment itself only through exclusive(), while the worker is idle.
class Stepper
{
public:
    // lockstep: wait for every snapshot to be fetched before the next step (for frame recording).
    Stepper(Experiment &experiment, bool playing, bool lockstep) : mExperiment(experiment), mPlaying(playing), mLockstep(lockstep), mThread([this] { run(); })
    {}
    ~Stepper()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mQuit = true;
        }
        mChanged.notify_all();
        mThread.join();
    }
    void setPlaying(bool playing)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mPlaying = playing;
        }
        mChanged.notify_all();
    }
    void requestStep()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mPendingSteps++;
        }
        mChanged.notify_all();
    }
    // Runs fn(experiment) once the worker is between steps, then publishes a fresh snapshot.
    template <class Fn>
    void exclusive(Fn &&fn)
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mChanged.wait(lock, [this] { return !mBusy; });
        fn(mExperiment);
        mLatest = toybox::Snapshot3D(mExperiment);
        mPublished++;
        lock.unlock();
        mChanged.notify_all();
    }
    // Moves the newest snapshot into `out` if one was published since the last fetch.
    bool fetch(ExperimentSnapshot &out)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mFetched == mPublished) return false;
            out = std::move(mLatest);
            mFetched = mPublished;
        }
        mChanged.notify_all();
        return true;
    }

private:
    Experiment &mExperiment;
    std::mutex mMutex;
    std::condition_variable mChanged;
    bool mPlaying;
    bool mLockstep;
    bool mQuit = false;
    bool mBusy = false;
    int mPendingSteps = 0;
    uint64_t mPublished = 0, mFetched = 0;
    ExperimentSnapshot mLatest;
    std::thread mThread;  // last, so it starts after every other member is initialized

    void run()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        while (true) {
            mChanged.wait(lock, [this] { return mQuit || ((mPlaying || mPendingSteps > 0) && !mExperiment.done() && (!mLockstep || mFetched == mPublished)); });
            if (mQuit) return;
            if (mPendingSteps > 0) mPendingSteps--;
            mBusy = true;
            lock.unlock();
            mExperiment.step();
            ExperimentSnapshot snapshot = toybox::Snapshot3D(mExperiment);
            lock.lock();
            mBusy = false;
            mLatest = std::move(snapshot);
            mPublished++;
            mChanged.notify_all();
        }
    }
};
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
        const Visibility visibility = options.visibility;
        const OrbitCamera camera = options.camera;
        if (!renderer.init(options.fontFile, error)) {
            std::cerr << "error: " << error << std::endl;
            status = 1;
        } else if (!StartRecorder(recorder, experiment.runs().size(), options) ||
                   !Capture(recorder, renderer, toybox::Snapshot3D(experiment), visibility, camera)) {
            status = 1;
        } else {
            while (experiment.step()) {
                if (!Capture(recorder, renderer, toybox::Snapshot3D(experiment), visibility, camera)) {
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
        Visibility visibility = options.visibility;
        OrbitCamera camera = options.camera;
        springls_gui::TriPlanarView slices;
        int slicedRun = -1;  // run shown in the slices (first adaptive run by default)
        bool paused = options.startPaused;
        bool summaryPrinted = false;
        bool resampling = true;
        int elementChoice = (experiment.options().elementType == ElementType::Quad) ? 0 : 1;
        int pacingChoice = static_cast<int>(experiment.pacing());  // the UI's copy; changes go through the stepper
        bool hasDual = false;
        for (const Run &run : experiment.runs()) {
            const Method3D m = run.as<toybox::Enright3DSimulation>()->method();
            hasDual |= (m == Method3D::Dweave || m == Method3D::AdaptiveDweave || m == Method3D::SemiLagrangianDweave);
        }
        enum class Drag {
            None,
            Rotate,
            Pan
        };
        Drag drag = Drag::None;
        ImVec2 lastDrag;
        // What is drawn: the newest snapshot from the stepper.
        ExperimentSnapshot shown = toybox::Snapshot3D(experiment);
        bool recording = false;
        if (!renderer.init(options.fontFile, error)) {
            std::cerr << "error: " << error << std::endl;
            status = 1;
        } else {
            recording = StartRecorder(recorder, shown.runs.size(), options) && Capture(recorder, renderer, shown, visibility, camera);
            if (!options.framesDir.empty() && !recording) status = 1;
        }
        Stepper stepper(experiment, !paused && status == 0, recording);
        const auto restart = [&]() {
            stepper.exclusive([&](Experiment &e) {
                std::string message;
                if (!e.restart(message)) std::cerr << "error: " << message << std::endl;
                e.forEachSimulation([&](toybox::Simulation &sim) { static_cast<toybox::Enright3DSimulation &>(sim).sim().setResamplingEnabled(resampling); });
            });
            summaryPrinted = false;
            if (stepper.fetch(shown) && recording) {
                recording = StartRecorder(recorder, shown.runs.size(), options) && Capture(recorder, renderer, shown, visibility, camera);
            }
        };
        while (status == 0 && !glfwWindowShouldClose(window)) {
            glfwPollEvents();
            if (stepper.fetch(shown) && recording && !Capture(recorder, renderer, shown, visibility, camera)) recording = false;
            if (!summaryPrinted && shown.done()) {
                stepper.exclusive([](Experiment &e) { e.printSummary(std::cout); });
                stepper.fetch(shown);
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
            const auto togglePause = [&]() {
                paused = !paused;
                stepper.setPlaying(!paused);
            };

            // Sidebar.
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(kSidebarWidth, static_cast<float>(winH)));
            ImGui::Begin("Enright 3D", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            if (ImGui::Button(paused ? "Play" : "Pause", ImVec2(80, 0))) togglePause();
            ImGui::SameLine();
            if (ImGui::Button("Step", ImVec2(80, 0))) stepper.requestStep();
            ImGui::SameLine();
            if (ImGui::Button("Restart", ImVec2(80, 0))) restart();
            if (ImGui::Button("Reset view")) camera.reset();
            if (shown.runs.size() > 1) {
                const char *pacings[] = {"Lockstep", "Equal compute", "Same sim time"};
                if (ImGui::Combo("pacing", &pacingChoice, pacings, IM_ARRAYSIZE(pacings))) {
                    stepper.exclusive([pacingChoice](Experiment &e) { e.setPacing(static_cast<Pacing>(pacingChoice)); });
                }
                ImGui::TextDisabled("M: cycle pacing");
            }
            ImGui::TextDisabled("Space: pause  R: reset view\nLeft drag: rotate  Right drag: pan\nScroll: zoom");
            if (recorder.active()) {
                ImGui::Separator();
                ImGui::Text("Recording %s", recording ? "on" : "stopped");
                ImGui::TextWrapped("%llu frames -> %s", static_cast<unsigned long long>(recorder.frameCount()), recorder.directory().string().c_str());
            }
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Checkbox("Re-sampling", &resampling)) {
                    stepper.exclusive([&](Experiment &e) {
                        e.forEachSimulation(
                            [&](toybox::Simulation &sim) { static_cast<toybox::Enright3DSimulation &>(sim).sim().setResamplingEnabled(resampling); });
                    });
                }
                ImGui::TextUnformatted("Springl elements (restarts):");
                const int previous = elementChoice;
                ImGui::RadioButton("Quads", &elementChoice, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Triangles", &elementChoice, 1);
                if (elementChoice != previous) {
                    stepper.exclusive([&](Experiment &e) { e.options().elementType = (elementChoice == 0) ? ElementType::Quad : ElementType::Triangle; });
                    restart();
                }
            }
            if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("IsoSurface", &visibility.isoSurface);
                ImGui::Checkbox(hasDual ? "Primal Springls" : "Springls", &visibility.primal);
                if (hasDual) ImGui::Checkbox("Dual Springls", &visibility.dual);
                ImGui::Checkbox("Particles", &visibility.particles);
                ImGui::Checkbox("Capsules", &visibility.capsules);
                // Leave room for the label to the right of the slider.
                ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("Capsule Radius").x + ImGui::GetStyle().ItemInnerSpacing.x));
                ImGui::SliderFloat("Capsule Radius", &visibility.capsuleRadius, 0.0f, 1.5f, "%.2f");
            }
            if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (const RunSnapshot &run : shown.runs) {
                    const double perStep = run.iteration > 0 ? 1000.0 * run.seconds / run.iteration : 0.0;
                    ImGui::Text("%-31s %7.1f ms/step", MethodTitle(run.method), perStep);
                    ImGui::Text("  %llu sub-steps, %.1f s", static_cast<unsigned long long>(run.substeps), run.seconds);
                }
                ImGui::Text("threads: %d", MaxThreads());
            }
            ImGui::End();

            // Right sidebar: tri-planar slices of one run's grid, colored by octree depth.
            if (!shown.runs.empty()) {
                if (slicedRun < 0 || slicedRun >= static_cast<int>(shown.runs.size())) {
                    slicedRun = 0;
                    for (size_t r = 0; r < shown.runs.size(); r++) {
                        if (shown.runs[r].grid) {
                            slicedRun = static_cast<int>(r);
                            break;
                        }
                    }
                }
                ImGui::SetNextWindowPos(ImVec2(std::max(kSidebarWidth, winW - kSliceSidebarWidth), 0));
                ImGui::SetNextWindowSize(ImVec2(kSliceSidebarWidth, static_cast<float>(winH)));
                ImGui::Begin("Grid slices", nullptr,
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);
                if (shown.runs.size() > 1) {
                    ImGui::SetNextItemWidth(-1.0f);
                    if (ImGui::BeginCombo("##run", MethodTitle(shown.runs[slicedRun].method))) {
                        for (size_t r = 0; r < shown.runs.size(); r++) {
                            if (ImGui::Selectable(MethodTitle(shown.runs[r].method), static_cast<int>(r) == slicedRun)) slicedRun = static_cast<int>(r);
                        }
                        ImGui::EndCombo();
                    }
                }
                const RunSnapshot &sliced = shown.runs[slicedRun];
                springls_gui::TriPlanarSource source;
                source.grid = sliced.grid;
                source.levelSet = sliced.levelSet;
                source.isoSurface = &sliced.manifold.isoSurface;
                const float D = sliced.config.voxelScale();
                source.extent = Vec3::Constant(D);
                source.defaultSlice = sliced.config.center * D;  // through the initial sphere
                source.maxDepth = sliced.maxDepth;
                source.baseCells = Vec3i::Constant(sliced.config.gridSize);
                source.identity = &sliced;
                source.stamp = shown.generation * 0x9E3779B97F4A7C15ull ^ sliced.iteration;
                slices.draw(source);
                ImGui::End();
            }

            // Orbit / pan / zoom on the panels, only for input ImGui does not want.
            const Rect view{kSidebarWidth, 0.0f, std::max(0.0f, winW - kSidebarWidth - kSliceSidebarWidth), static_cast<float>(winH)};
            const std::vector<Rect> panels = Renderer::Layout(shown.runs.size(), view);
            const ImVec2 cursor = io.MousePos;
            bool hovered = false;
            for (const Rect &p : panels) hovered |= p.contains(cursor.x, cursor.y);
            if (!io.WantCaptureMouse && hovered) {
                if (io.MouseWheel != 0.0f) camera.zoom = std::clamp(camera.zoom * (1.0f + kZoomPerScrollTick * io.MouseWheel), kMinZoom, kMaxZoom);
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                    drag = io.KeyShift ? Drag::Pan : Drag::Rotate;
                    lastDrag = cursor;
                } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                    drag = Drag::Pan;
                    lastDrag = cursor;
                }
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Right) && !ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                drag = Drag::None;
            }
            if (drag != Drag::None && !panels.empty() && panels[0].w > 0.0f) {
                const float dx = cursor.x - lastDrag.x, dy = cursor.y - lastDrag.y;
                if (drag == Drag::Rotate) {
                    camera.rotate(dx, dy);
                } else {
                    // The panel spans 2 * 0.75 / zoom unit-cube units.
                    const float unitsPerPixel = 1.5f / camera.zoom / panels[0].w;
                    camera.panX += dx * unitsPerPixel;
                    camera.panY -= dy * unitsPerPixel;
                }
                lastDrag = cursor;
            }
            if (!io.WantCaptureKeyboard) {
                if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) togglePause();
                if (ImGui::IsKeyPressed(ImGuiKey_R, false)) camera.reset();
                if (ImGui::IsKeyPressed(ImGuiKey_M, false) && shown.runs.size() > 1) {
                    pacingChoice = (pacingChoice + 1) % 3;
                    stepper.exclusive([pacingChoice](Experiment &e) { e.setPacing(static_cast<Pacing>(pacingChoice)); });
                }
            }

            ImGui::Render();
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbW, fbH);
            glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            renderer.draw(shown, visibility, camera, view, static_cast<float>(winW), static_cast<float>(winH), pixelRatio,
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
}  // namespace enright3d
