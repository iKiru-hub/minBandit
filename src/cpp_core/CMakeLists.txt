cmake_minimum_required(VERSION 3.15)
project(mblib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add pybind11
add_subdirectory(extern/pybind11)

# Header files
set(HEADERS
    include/core.hpp
    include/utils.hpp
)

# Python module
set(MODULE_SOURCES
    src/binding.cpp
)

# Add the pybind11 module
pybind11_add_module(mblib ${MODULE_SOURCES})

# Test executable
set(TEST_SOURCES
    src/test.cpp
)

# Add the test executable
add_executable(core_test ${TEST_SOURCES})

# Include directories for both targets
target_include_directories(mblib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(core_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries for both targets
target_link_libraries(mblib
    PRIVATE
        Eigen3::Eigen
)

target_link_libraries(core_test
    PRIVATE
        Eigen3::Eigen
)

# Set compiler flags for macOS
if(APPLE)
    # Flags for Python module
    target_compile_options(mblib PRIVATE 
        -stdlib=libc++ 
        -O3
        -Wall
        -Wextra
        -mmacosx-version-min=10.9
    )
    set_target_properties(mblib PROPERTIES
        LINK_FLAGS "-stdlib=libc++ -mmacosx-version-min=10.9"
    )
    
    # Flags for test executable
    target_compile_options(core_test PRIVATE 
        -stdlib=libc++ 
        -O3
        -Wall
        -Wextra
        -mmacosx-version-min=10.9
    )
    set_target_properties(core_test PROPERTIES
        LINK_FLAGS "-stdlib=libc++ -mmacosx-version-min=10.9"
    )
endif()


# suppress warnings
add_definitions(-w)

# print that the warning suppression is on
message(">>> Yo sup dawg, beware that warning suppression is on <<<")

