// One registry entry's program (TOYBOX_PROGRAM, e.g. "enright-2d-adaptive-multi-dweave"): toybox with that
// entry selected. CMakeLists.txt builds it once per entry (TOYBOX_PROGRAMS); see toybox --list.
#include <common/Toybox.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
    const toybox::Demo *demo = toybox::demos::Find(TOYBOX_PROGRAM);
    if (!demo) {
        std::cerr << "error: " << TOYBOX_PROGRAM << " is not a registry entry's program" << std::endl;
        return 2;
    }
    return toybox::Main(argc, argv, *demo);
}
