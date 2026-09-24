# libmeepmeep: the MeepMeep Taylor-series orbit evaluators as a C99 library.
#
# The library compiles the sources shared with the OpenCL backend
# (../meepmeep/backends/opencl/*.cl) as plain C, so it lives inside the
# Python package's repository but is built and installed independently of
# it. Typical use:
#
#     cmake -S c -B c/build -DCMAKE_INSTALL_PREFIX=$HOME/.local
#     cmake --build c/build
#     cmake --install c/build
#
# Then `#include <meepmeep.h>` and link with -lmeepmeep -lm.

cmake_minimum_required(VERSION 3.15)
project(meepmeep VERSION 1.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_SHARED_LIBS "Build libmeepmeep as a shared library" ON)
option(MEEPMEEP_BUILD_EXAMPLES "Build the example programs" OFF)

set(MEEPMEEP_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../meepmeep/backends/opencl"
    CACHE PATH "Directory holding the .cl sources shared with the OpenCL backend")

add_library(meepmeep
    src/meepmeep.c
    src/expansion_points.c
    src/orbit.c)

target_include_directories(meepmeep
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${MEEPMEEP_SHARED_SOURCE_DIR})

# Strict IEEE math is part of the contract: the numba twins that are
# compiled without fastmath rely on it, and the parity tests pin ~1e-12
# agreement. Do not add -ffast-math or -Ofast.
target_compile_options(meepmeep PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wno-unused-parameter -fno-fast-math>)

find_library(MEEPMEEP_LIBM m)
if(MEEPMEEP_LIBM)
    target_link_libraries(meepmeep PUBLIC ${MEEPMEEP_LIBM})
endif()

set_target_properties(meepmeep PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER include/meepmeep.h)

include(GNUInstallDirs)
install(TARGETS meepmeep
    EXPORT meepmeepTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT meepmeepTargets
    NAMESPACE meepmeep::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meepmeep)

if(MEEPMEEP_BUILD_EXAMPLES)
    add_executable(transit examples/transit.c)
    target_link_libraries(transit PRIVATE meepmeep)
endif()
