cmake_minimum_required(VERSION 3.20)
# Default to Release before project(), which otherwise sets Debug for MSVC-like compilers.
get_property(SPRINGLS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT CMAKE_BUILD_TYPE AND NOT SPRINGLS_MULTI_CONFIG)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()
project(springls LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPRINGLS_USE_OPENMP "Parallelize with OpenMP when available" ON)
option(SPRINGLS_BUILD_GUI "Build the GUI (GLFW + OpenGL 3 + Dear ImGui + NanoVG)" ON)
option(SPRINGLS_BUILD_EXAMPLES "Build toybox and one program per implemented configuration (enright-2d-springls, ..., mesh-3d-dweave)" ON)
option(SPRINGLS_BUILD_TESTS "Build the unit tests" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>")
endif()
if(MSVC)
	# Compile each target's sources in parallel (/MP), read sources as UTF-8, and lift the object-section limit
	# that large Eigen-heavy translation units can hit (/bigobj). /MP only helps the Visual Studio generator.
	option(SPRINGLS_MSVC_MP "Compile with /MP (parallel compilation within a target)" ON)
	add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8;/bigobj>")
	if(SPRINGLS_MSVC_MP AND CMAKE_GENERATOR MATCHES "Visual Studio")
		add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/MP>")
	endif()
	# M_PI and friends from <cmath>; min/max macros from <windows.h> off.
	add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

# ------------------------------------------------------------------------------------------------
# Core library: spring level sets, DoubleWeave, adaptive variants. Depends only on Eigen (+ OpenMP).
# ------------------------------------------------------------------------------------------------
# Eigen (header only, from the submodule at 3.4.0); no installed copy is used.
set(SPRINGLS_EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
if(NOT EXISTS ${SPRINGLS_EIGEN_DIR}/Eigen/Core)
	message(FATAL_ERROR "Eigen not found in external/eigen. Run 'git submodule update --init external/eigen'.")
endif()
# kdtree-cpp (header only, from the submodule): its BoxTree locates the nearest triangle in MeshDistance3D. Only its
# headers are used; its own CMake project (tests that fetch GoogleTest) is not added.
set(SPRINGLS_KDTREE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/kdtree-cpp/src)
if(NOT EXISTS ${SPRINGLS_KDTREE_DIR}/boxtree.hpp)
	message(FATAL_ERROR "kdtree-cpp not found in external/kdtree-cpp. Run 'git submodule update --init external/kdtree-cpp'.")
endif()
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/happly/happly.h)
	message(FATAL_ERROR "happly not found in external/happly. Run 'git submodule update --init external/happly'.")
endif()
# tinyvdb (from the submodule) writes and reads the adaptive level set as OpenVDB files. Only its I/O library
# (tinyvdb) and the sparse-to-tree builder in tinyvdb_mesh_ops are used; examples, tests, the GPU backend and
# its own submodules are not needed.
set(SPRINGLS_TINYVDB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyvdb)
if(NOT EXISTS ${SPRINGLS_TINYVDB_DIR}/src/tinyvdb_io.h)
	message(FATAL_ERROR "tinyvdb not found in external/tinyvdb. Run 'git submodule update --init external/tinyvdb'.")
endif()
set(TINYVDB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYVDB_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(TINYVDB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYVDB_BUILD_VDBRENDER OFF CACHE BOOL "" FORCE)
set(TINYVDB_BUILD_TVDBVIEW OFF CACHE BOOL "" FORCE)
set(TINYVDB_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(TINYVDB_BUILD_GPU OFF CACHE BOOL "" FORCE)
set(TINYVDB_USE_CCACHE OFF CACHE BOOL "" FORCE)
set(TINYVDB_THREAD_BACKEND NONE CACHE STRING "" FORCE)  # its task pool is not used
# Portable binaries: no AVX2 / F16C requirement.
set(TINYVDB_SIMD OFF CACHE BOOL "" FORCE)
add_subdirectory(${SPRINGLS_TINYVDB_DIR} ${CMAKE_BINARY_DIR}/tinyvdb EXCLUDE_FROM_ALL)

add_library(springls_eigen INTERFACE)
target_include_directories(springls_eigen SYSTEM INTERFACE ${SPRINGLS_EIGEN_DIR})

file(GLOB SPRINGLS_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp)
add_library(springls STATIC ${SPRINGLS_SOURCES})
target_include_directories(springls PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# cereal (header only, from the submodule) writes checkpoints; only the library's sources use it.
target_include_directories(springls PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/cereal/include)
target_include_directories(springls PRIVATE ${SPRINGLS_KDTREE_DIR})
# happly (header only, from the submodule) reads and writes PLY files; only the library's sources use it.
target_include_directories(springls PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/happly)
target_link_libraries(springls PUBLIC springls_eigen)
target_link_libraries(springls PRIVATE tinyvdb_mesh_ops tinyvdb)

if(SPRINGLS_USE_OPENMP)
	# MSVC's default OpenMP 2.0 lacks min/max reductions; use its LLVM runtime (CMake 3.30+).
	if(MSVC AND NOT DEFINED OpenMP_RUNTIME_MSVC)
		set(OpenMP_RUNTIME_MSVC llvm)
	endif()
	find_package(OpenMP COMPONENTS CXX)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(springls PUBLIC OpenMP::OpenMP_CXX)
		target_compile_definitions(springls PUBLIC SPRINGLS_OPENMP)
		message(STATUS "springls: OpenMP enabled")
	else()
		message(STATUS "springls: OpenMP not found, building single-threaded")
	endif()
endif()

# ------------------------------------------------------------------------------------------------
# toybox core: configurations, registry, simulation adapters, experiment, command line. No GL.
# ------------------------------------------------------------------------------------------------
# argparse (header only, from the submodule).
add_library(argparse INTERFACE)
target_include_directories(argparse INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/argparse/include)

file(GLOB SPRINGLS_TOYBOX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apps/core/*.cpp)
add_library(springls_toybox STATIC ${SPRINGLS_TOYBOX_SOURCES})
target_include_directories(springls_toybox PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps)
target_link_libraries(springls_toybox PUBLIC springls springls_intrinsic)
target_link_libraries(springls_toybox PRIVATE argparse)
# Defaults of the command line: the panel font, the mesh scenario's assets and its cache of initial states.
target_compile_definitions(springls_toybox PRIVATE
	SPRINGLS_FONT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/example/Roboto-Regular.ttf"
	SPRINGLS_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
	SPRINGLS_CACHE_DIR="${CMAKE_BINARY_DIR}/cache")

# ------------------------------------------------------------------------------------------------
# intrinsic-springls runner (headless): multi-object springls on a triangle-mesh surface.
# ------------------------------------------------------------------------------------------------
add_library(springls_intrinsic STATIC ${CMAKE_CURRENT_SOURCE_DIR}/apps/intrinsic-springls/Runner.cpp ${CMAKE_CURRENT_SOURCE_DIR}/apps/intrinsic-springls/Render.cpp)
# stb_image_write (header only, from nanovg) writes the PNG views.
target_include_directories(springls_intrinsic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/src/thirdParty)
target_include_directories(springls_intrinsic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps)
target_link_libraries(springls_intrinsic PUBLIC springls)
target_link_libraries(springls_intrinsic PRIVATE argparse)
target_compile_definitions(springls_intrinsic PRIVATE SPRINGLS_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")

# Unit tests: every tests/*.cpp is part of the springls-tests executable (a tiny CHECK/TEST harness).
if(SPRINGLS_BUILD_TESTS)
	enable_testing()
	file(GLOB SPRINGLS_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
	add_executable(springls-tests ${SPRINGLS_TEST_SOURCES})
	target_link_libraries(springls-tests PRIVATE springls springls_toybox springls_intrinsic)
	# Tests that compare with the captured baselines (build/baseline*) or read the README find them here.
	target_compile_definitions(springls-tests PRIVATE SPRINGLS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
	add_test(NAME springls-tests COMMAND springls-tests)
endif()

if(NOT SPRINGLS_BUILD_EXAMPLES)
	return()
endif()

# ------------------------------------------------------------------------------------------------
# GUI dependencies: GLFW, OpenGL, Dear ImGui, NanoVG (C++ port).
# ------------------------------------------------------------------------------------------------
if(SPRINGLS_BUILD_GUI)
	find_package(OpenGL REQUIRED)
	find_package(glfw3 3.3 CONFIG QUIET)
	if(NOT TARGET glfw)
		set(SPRINGLS_GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/external/glfw)
		if(EXISTS ${SPRINGLS_GLFW_DIR}/CMakeLists.txt)
			set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
			set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
			set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
			set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
			add_subdirectory(${SPRINGLS_GLFW_DIR} ${CMAKE_BINARY_DIR}/glfw EXCLUDE_FROM_ALL)
		else()
			message(FATAL_ERROR "GLFW not found. Install it (e.g. libglfw3-dev) or run "
				"'git -C external/nanovg submodule update --init external/glfw', or set SPRINGLS_BUILD_GUI=OFF.")
		endif()
	endif()

	set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
	add_library(imgui STATIC
		${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_tables.cpp
		${IMGUI_DIR}/imgui_widgets.cpp ${IMGUI_DIR}/imgui_demo.cpp
		${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
	target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
	target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

	set(NANOVG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/src)
	file(GLOB NANOVG_SOURCES CONFIGURE_DEPENDS ${NANOVG_DIR}/nvg/*.cpp)
	add_library(nanovg STATIC ${NANOVG_SOURCES})
	target_include_directories(nanovg PUBLIC ${NANOVG_DIR} ${NANOVG_DIR}/nvg ${NANOVG_DIR}/nvg/stable ${NANOVG_DIR}/thirdParty)
	target_compile_definitions(nanovg PRIVATE NANOVG_FAST_SQRT=1)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		target_compile_options(nanovg PRIVATE -w)
	endif()

	# NanoVG's Vulkan backend (nanovg_vk.hpp, header only, plus the stable C API in nanovg_vk), built
	# against the Vulkan headers pinned in nanovg's submodule. The GUI still draws with the GL3 backend;
	# this keeps the Vulkan backend compiling for the coming port.
	set(SPRINGLS_VULKAN_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/external/Vulkan-Headers)
	if(NOT EXISTS ${SPRINGLS_VULKAN_HEADERS_DIR}/CMakeLists.txt)
		message(FATAL_ERROR "Vulkan headers not found in ${SPRINGLS_VULKAN_HEADERS_DIR}. Run "
			"'git -C external/nanovg submodule update --init external/Vulkan-Headers', or set SPRINGLS_BUILD_GUI=OFF.")
	endif()
	add_subdirectory(${SPRINGLS_VULKAN_HEADERS_DIR} ${CMAKE_BINARY_DIR}/Vulkan-Headers EXCLUDE_FROM_ALL)
	add_library(nanovg_vk STATIC ${NANOVG_DIR}/nvg/stable/nanovg_vk.cpp)
	target_link_libraries(nanovg_vk PUBLIC nanovg Vulkan::Headers)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		target_compile_options(nanovg_vk PRIVATE -w)
	endif()

	# OpenGL 3 entry points: on Windows the glad loader (header only) bundled in GLFW's deps, otherwise
	# the system prototypes (Linux / macOS).
	set(SPRINGLS_GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/external/glfw/deps)
	if(WIN32 AND NOT EXISTS ${SPRINGLS_GLAD_DIR}/glad/gl.h)
		message(FATAL_ERROR "glad not found in ${SPRINGLS_GLAD_DIR}. Run "
			"'git -C external/nanovg submodule update --init external/glfw', or set SPRINGLS_BUILD_GUI=OFF.")
	endif()
endif()

# ------------------------------------------------------------------------------------------------
# toybox windows: the 2D Enright, 3D Enright, mesh and sphere windows, reading the core's runs.
# ------------------------------------------------------------------------------------------------
set(TOYBOX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apps)
if(SPRINGLS_BUILD_GUI)
	file(GLOB TOYBOX_DEMO_SOURCES CONFIGURE_DEPENDS
		${TOYBOX_DIR}/common/*.cpp ${TOYBOX_DIR}/enright2d/*.cpp ${TOYBOX_DIR}/enright3d/*.cpp ${TOYBOX_DIR}/mesh/*.cpp ${TOYBOX_DIR}/sphere/*.cpp)
	add_library(springls_toybox_gui STATIC ${TOYBOX_DEMO_SOURCES})
	target_include_directories(springls_toybox_gui PUBLIC ${TOYBOX_DIR}/common)
	target_include_directories(springls_toybox_gui PRIVATE ${NANOVG_DIR}/thirdParty)
	target_link_libraries(springls_toybox_gui PUBLIC springls_toybox imgui nanovg glfw OpenGL::GL)
	target_compile_definitions(springls_toybox_gui PUBLIC SPRINGLS_GUI)
	# The mesh window's matcaps.
	target_compile_definitions(springls_toybox_gui PRIVATE SPRINGLS_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
	if(WIN32)
		target_include_directories(springls_toybox_gui PUBLIC ${SPRINGLS_GLAD_DIR})
		target_compile_definitions(springls_toybox_gui PUBLIC SPRINGLS_GLAD)
	else()
		target_compile_definitions(springls_toybox_gui PUBLIC GL_GLEXT_PROTOTYPES=1)
	endif()
endif()

# Links an executable of the toybox (it calls toybox::Main from common/Toybox.hpp) to the core and, in GUI builds, the windows.
function(springls_toybox_executable target)
	add_executable(${target} ${ARGN})
	if(SPRINGLS_BUILD_GUI)
		target_link_libraries(${target} PRIVATE springls_toybox_gui)
	else()
		target_link_libraries(${target} PRIVATE springls_toybox)
	endif()
endfunction()

springls_toybox_executable(toybox ${TOYBOX_DIR}/Toybox.cpp)

add_executable(intrinsic-springls ${CMAKE_CURRENT_SOURCE_DIR}/apps/intrinsic-springls/main.cpp)
target_link_libraries(intrinsic-springls PRIVATE springls_intrinsic)

# ------------------------------------------------------------------------------------------------
# One program per registry entry, <scenario>-<domain>-<method key>, in build/demo (toybox --list
# prints them; a test checks this list against the registry).
# ------------------------------------------------------------------------------------------------
set(TOYBOX_PROGRAMS
	enright-2d-springls enright-2d-adaptive-springls enright-2d-dweave enright-2d-adaptive-dweave enright-2d-levelset enright-2d-adaptive-levelset
	enright-2d-semi-lagrangian-dweave
	enright-2d-multi-levelset enright-2d-adaptive-multi-levelset enright-2d-multi-springls enright-2d-adaptive-multi-springls enright-2d-multi-dweave
	enright-2d-adaptive-multi-dweave
	enright-3d-springls enright-3d-adaptive-springls enright-3d-dweave enright-3d-adaptive-dweave enright-3d-semi-lagrangian-dweave
	enright-3d-levelset enright-3d-adaptive-levelset
	mesh-3d-dweave
	sphere-surf-multi-springls stone-surf-multi-springls
)
foreach(PROGRAM ${TOYBOX_PROGRAMS})
	springls_toybox_executable(${PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/apps/Demo.cpp)
	target_compile_definitions(${PROGRAM} PRIVATE TOYBOX_PROGRAM="${PROGRAM}")
	# The programs go in build/demo, apart from toybox and the tests.
	set_target_properties(${PROGRAM} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)
endforeach()
