#pragma once
// What every executable calls: `return toybox::Main(argc, argv, toybox::demos::Enright2D());`. GUI builds
// (SPRINGLS_GUI) pass the windows to the core's Main; headless builds pass none.
#include <core/Demos.hpp>
#include <core/Main.hpp>
#ifdef SPRINGLS_GUI
#include <common/Window.hpp>
#endif

namespace toybox
{
inline int Main(int argc, char *argv[], const Demo &demo)
{
#ifdef SPRINGLS_GUI
    return Main(argc, argv, demo, &RunWindow);
#else
    return Main(argc, argv, demo, nullptr);
#endif
}
}  // namespace toybox
