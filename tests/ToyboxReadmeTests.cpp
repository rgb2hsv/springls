#include "Check.hpp"

#include <core/Registry.hpp>

#include <fstream>
#include <sstream>
#include <string>

// README.md carries `toybox --list` verbatim, so the matrix cannot drift from the registry.
TEST(readme_matrix_matches_registry)
{
    std::ifstream in(std::string(SPRINGLS_SOURCE_DIR) + "/README.md");
    std::stringstream text;
    text << in.rdbuf();
    const std::string matrix = toybox::Registry::Instance().markdownMatrix();
    CHECK(text.str().find(matrix) != std::string::npos);
}
