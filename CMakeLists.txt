cmake_minimum_required(VERSION 3.20)
project(springls LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SPRINGLS_USE_OPENMP "Parallelize with OpenMP when available" ON)
option(SPRINGLS_BUILD_GUI "Build the GUI (GLFW + OpenGL 3 + Dear ImGui + NanoVG)" ON)
option(SPRINGLS_BUILD_EXAMPLES "Build the example programs" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>")
endif()

# ------------------------------------------------------------------------------------------------
# Core library: spring level sets, doubleweave, adaptive variants. Depends only on Eigen (+ OpenMP).
# ------------------------------------------------------------------------------------------------
# Eigen (header only, from the submodule at 3.4.0); no installed copy is used.
set(SPRINGLS_EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)
if(NOT EXISTS ${SPRINGLS_EIGEN_DIR}/Eigen/Core)
	message(FATAL_ERROR "Eigen not found in external/eigen. Run 'git submodule update --init external/eigen'.")
endif()
add_library(springls_eigen INTERFACE)
target_include_directories(springls_eigen SYSTEM INTERFACE ${SPRINGLS_EIGEN_DIR})

file(GLOB SPRINGLS_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/springls/*.cpp)
add_library(springls STATIC ${SPRINGLS_SOURCES})
target_include_directories(springls PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# cereal (header only, from the submodule) writes checkpoints; only the library's sources use it.
target_include_directories(springls PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/cereal/include)
target_link_libraries(springls PUBLIC springls_eigen)

if(SPRINGLS_USE_OPENMP)
	# MSVC's default OpenMP 2.0 lacks min/max reductions and tasks; use its LLVM runtime (CMake 3.30+).
	set(OpenMP_RUNTIME_MSVC llvm)
	find_package(OpenMP COMPONENTS CXX)
	if(OpenMP_CXX_FOUND)
		target_link_libraries(springls PUBLIC OpenMP::OpenMP_CXX)
		target_compile_definitions(springls PUBLIC SPRINGLS_OPENMP)
		message(STATUS "springls: OpenMP enabled")
	else()
		message(STATUS "springls: OpenMP not found, building single-threaded")
	endif()
endif()

if(NOT SPRINGLS_BUILD_EXAMPLES)
	return()
endif()

# argparse (header only, from the submodule).
add_library(argparse INTERFACE)
target_include_directories(argparse INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/argparse/include)

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
	target_include_directories(nanovg PUBLIC ${NANOVG_DIR} ${NANOVG_DIR}/nvg ${NANOVG_DIR}/nvg/legacy ${NANOVG_DIR}/thirdParty)
	target_compile_definitions(nanovg PRIVATE NANOVG_FAST_SQRT=1)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		target_compile_options(nanovg PRIVATE -w)
	endif()

	# OpenGL 3 entry points: GLEW where available (needed on Windows), otherwise the system
	# prototypes (Linux / macOS).
	find_package(GLEW QUIET)
	if(WIN32 AND NOT GLEW_FOUND)
		# No installed GLEW: build the static library from nanovg's submodule.
		set(SPRINGLS_GLEW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/external/glew)
		if(EXISTS ${SPRINGLS_GLEW_DIR}/src/glew.c)
			add_library(springls_glew STATIC ${SPRINGLS_GLEW_DIR}/src/glew.c)
			target_include_directories(springls_glew PUBLIC ${SPRINGLS_GLEW_DIR}/include)
			target_compile_definitions(springls_glew PUBLIC GLEW_STATIC)
			target_link_libraries(springls_glew PUBLIC OpenGL::GL)
			add_library(GLEW::GLEW ALIAS springls_glew)
			set(GLEW_FOUND TRUE)
		else()
			message(FATAL_ERROR "GLEW not found. On Windows run "
				"'git -C external/nanovg submodule update --init external/glew', or set SPRINGLS_BUILD_GUI=OFF.")
		endif()
	endif()
endif()

# ------------------------------------------------------------------------------------------------
# Enright 2D example.
# ------------------------------------------------------------------------------------------------
set(ENRIGHT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/enright2d)
set(ENRIGHT_SOURCES ${ENRIGHT_DIR}/main.cpp ${ENRIGHT_DIR}/Experiment.cpp)
if(SPRINGLS_BUILD_GUI)
	list(APPEND ENRIGHT_SOURCES ${ENRIGHT_DIR}/Renderer.cpp ${ENRIGHT_DIR}/Gui.cpp)
endif()
add_executable(enright2d ${ENRIGHT_SOURCES})
target_link_libraries(enright2d PRIVATE springls argparse)
if(SPRINGLS_BUILD_GUI)
	target_link_libraries(enright2d PRIVATE imgui nanovg glfw OpenGL::GL)
	target_compile_definitions(enright2d PRIVATE SPRINGLS_GUI
		SPRINGLS_FONT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/example/Roboto-Regular.ttf")
	if(GLEW_FOUND AND WIN32)
		target_link_libraries(enright2d PRIVATE GLEW::GLEW)
		target_compile_definitions(enright2d PRIVATE SPRINGLS_GLEW)
	elseif(NOT WIN32)
		target_compile_definitions(enright2d PRIVATE GL_GLEXT_PROTOTYPES=1)
	endif()
endif()

# ------------------------------------------------------------------------------------------------
# Semi-Lagrangian vs CFL-limited doubleweave (reuses the Enright 2D experiment, renderer and GUI).
# ------------------------------------------------------------------------------------------------
set(SL_ENRIGHT2D_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/examples/sl-enright2d/main.cpp ${ENRIGHT_DIR}/Experiment.cpp)
if(SPRINGLS_BUILD_GUI)
	list(APPEND SL_ENRIGHT2D_SOURCES ${ENRIGHT_DIR}/Renderer.cpp ${ENRIGHT_DIR}/Gui.cpp)
endif()
add_executable(sl-enright2d ${SL_ENRIGHT2D_SOURCES})
target_include_directories(sl-enright2d PRIVATE ${ENRIGHT_DIR})
target_link_libraries(sl-enright2d PRIVATE springls argparse)
if(SPRINGLS_BUILD_GUI)
	target_link_libraries(sl-enright2d PRIVATE imgui nanovg glfw OpenGL::GL)
	target_compile_definitions(sl-enright2d PRIVATE SPRINGLS_GUI
		SPRINGLS_FONT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/example/Roboto-Regular.ttf")
	if(GLEW_FOUND AND WIN32)
		target_link_libraries(sl-enright2d PRIVATE GLEW::GLEW)
		target_compile_definitions(sl-enright2d PRIVATE SPRINGLS_GLEW)
	elseif(NOT WIN32)
		target_compile_definitions(sl-enright2d PRIVATE GL_GLEXT_PROTOTYPES=1)
	endif()
endif()

# ------------------------------------------------------------------------------------------------
# Enright 3D example.
# ------------------------------------------------------------------------------------------------
set(ENRIGHT3D_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/enright3d)
set(ENRIGHT3D_SOURCES ${ENRIGHT3D_DIR}/main.cpp ${ENRIGHT3D_DIR}/Experiment.cpp)
if(SPRINGLS_BUILD_GUI)
	list(APPEND ENRIGHT3D_SOURCES ${ENRIGHT3D_DIR}/Renderer.cpp ${ENRIGHT3D_DIR}/Gui.cpp ${ENRIGHT3D_DIR}/TriPlanar.cpp)
endif()
add_executable(enright3d ${ENRIGHT3D_SOURCES})
target_link_libraries(enright3d PRIVATE springls argparse)
if(SPRINGLS_BUILD_GUI)
	target_link_libraries(enright3d PRIVATE imgui nanovg glfw OpenGL::GL)
	target_compile_definitions(enright3d PRIVATE SPRINGLS_GUI
		SPRINGLS_FONT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/example/Roboto-Regular.ttf")
	if(GLEW_FOUND AND WIN32)
		target_link_libraries(enright3d PRIVATE GLEW::GLEW)
		target_compile_definitions(enright3d PRIVATE SPRINGLS_GLEW)
	elseif(NOT WIN32)
		target_compile_definitions(enright3d PRIVATE GL_GLEXT_PROTOTYPES=1)
	endif()
endif()

# ------------------------------------------------------------------------------------------------
# Semi-Lagrangian vs CFL-limited doubleweave in 3D (reuses the Enright 3D experiment, renderer and GUI).
# ------------------------------------------------------------------------------------------------
set(SL_ENRIGHT3D_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/examples/sl-enright3d/main.cpp ${ENRIGHT3D_DIR}/Experiment.cpp)
if(SPRINGLS_BUILD_GUI)
	list(APPEND SL_ENRIGHT3D_SOURCES ${ENRIGHT3D_DIR}/Renderer.cpp ${ENRIGHT3D_DIR}/Gui.cpp ${ENRIGHT3D_DIR}/TriPlanar.cpp)
endif()
add_executable(sl-enright3d ${SL_ENRIGHT3D_SOURCES})
target_include_directories(sl-enright3d PRIVATE ${ENRIGHT3D_DIR})
target_link_libraries(sl-enright3d PRIVATE springls argparse)
if(SPRINGLS_BUILD_GUI)
	target_link_libraries(sl-enright3d PRIVATE imgui nanovg glfw OpenGL::GL)
	target_compile_definitions(sl-enright3d PRIVATE SPRINGLS_GUI
		SPRINGLS_FONT_FILE="${CMAKE_CURRENT_SOURCE_DIR}/external/nanovg/example/Roboto-Regular.ttf")
	if(GLEW_FOUND AND WIN32)
		target_link_libraries(sl-enright3d PRIVATE GLEW::GLEW)
		target_compile_definitions(sl-enright3d PRIVATE SPRINGLS_GLEW)
	elseif(NOT WIN32)
		target_compile_definitions(sl-enright3d PRIVATE GL_GLEXT_PROTOTYPES=1)
	endif()
endif()
