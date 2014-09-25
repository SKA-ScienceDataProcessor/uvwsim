project(uvwsim C CXX)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
if ("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" VERSION_GREATER 2.8)
    cmake_policy(SET CMP0042 OLD)
endif()

set(UVWSIM_VERSION_MAJOR 1)
set(UVWSIM_VERSION_MINOR 0)
set(UVWSIM_VERSION "${UVWSIM_VERSION_MAJOR}.${UVWSIM_VERSION_MINOR}")

# Add the local cmake directory into the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

include(compiler_options)

# Build the library.
set(libname uvwsim)
include_directories(${PROJECT_SOURCE_DIR}/src)
add_library(${libname} src/uvwsim.c)
#set_source_files_properties(src/uvwsim.c PROPERTIES LANGUAGE CXX)
install(TARGETS ${libname} DESTINATION ${UVWSIM_LIB_INSTALL_DIR})
install(FILES src/uvwsim.h DESTINATION ${UVWSIM_INCLUDE_INSTALL_DIR})

# Build the python interface (if python [2.7] and numpy [1.8] is found)
# add_subdirectory(python)

# Build unit tests
enable_testing()
add_subdirectory(test)
