cmake_minimum_required(VERSION 3.15)
project(mblib LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add pybind11
add_subdirectory(extern/pybind11)

# Headers and Sources
set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.hpp
)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/binding.cpp
)

# Test Sources
set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp
)

# Pybind11 module
pybind11_add_module(mblib ${SOURCES})
target_include_directories(mblib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mblib PRIVATE Eigen3::Eigen)

# Test executable
add_executable(core_test ${TEST_SOURCES})
target_include_directories(core_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(core_test PRIVATE Eigen3::Eigen)

# Shared compiler flags
set(COMMON_FLAGS -O3 -Wall -Wextra)
if(APPLE)
    list(APPEND COMMON_FLAGS -stdlib=libc++ -mmacosx-version-min=10.9)
endif()

# Apply flags
target_compile_options(mblib PRIVATE ${COMMON_FLAGS})
target_compile_options(core_test PRIVATE ${COMMON_FLAGS})

# Suppress warnings globally (if needed)
add_definitions(-w)

# Print a warning suppression message
message(">>> Warning suppression is ENABLED. Proceed with caution. <<<")
