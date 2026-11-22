// intrinsic-springls: single-region springls on a triangle-mesh surface (headless). See README.md.
#include <intrinsic-springls/Runner.hpp>

#include <iostream>

int main(int argc, char **argv)
{
    using namespace springls::intrinsic;
    RunnerOptions options;
    std::string error;
    bool help = false;
    if (!ParseArguments(argc, argv, options, error, &help)) {
        (help ? std::cout : std::cerr) << error << std::endl;
        return help ? 0 : 2;
    }
    RunnerResult result;
    if (!Run(options, result, error, &std::cout)) {
        std::cerr << "intrinsic-springls: " << error << std::endl;
        return 1;
    }
    return 0;
}
