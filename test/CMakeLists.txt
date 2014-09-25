# Set include directories for google testing framework.
include_directories(${PROJECT_SOURCE_DIR}/test/gtest-1.7.0)
include_directories(${PROJECT_SOURCE_DIR}/test/gtest-1.7.0/include)

# Build google testing framework.
add_subdirectory(gtest-1.7.0)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_uvwsim.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/_test_uvwsim.cpp @ONLY)
set(name test_uvwsim)
add_executable(${name} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/_test_uvwsim.cpp)
target_link_libraries(${name} uvwsim gtest_main)
add_test(${name} ${name})

include(copy_files)
copy_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/data/VLA_A_hor_xyz.txt
    ${CMAKE_CURRENT_BINARY_DIR}/VLA_A_hor_xyz.txt
)

# Configure python tests - if python bindings are being built.
if (UVWSIM_PYTHON_API)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test_python_interface.py
        ${CMAKE_CURRENT_BINARY_DIR}/test_python_interface.py @ONLY)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/run_python_tests.sh 
        ${CMAKE_CURRENT_BINARY_DIR}/run_python_tests.sh @ONLY)
endif()

# Enable to copy CASA config files (if available)
#copy_file(
#    ${CMAKE_CURRENT_SOURCE_DIR}/data/vla.a.cfg
#    ${CMAKE_CURRENT_BINARY_DIR}/vla.a.cfg
#)
#copy_file(
#    ${CMAKE_CURRENT_SOURCE_DIR}/data/WSRT.cfg
#    ${CMAKE_CURRENT_BINARY_DIR}/WSRT.cfg
#)
#copy_file(
#    ${CMAKE_CURRENT_SOURCE_DIR}/data/meerkat.cfg
#    ${CMAKE_CURRENT_BINARY_DIR}/meerkat.cfg
#)
