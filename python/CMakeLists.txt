find_package(PythonLibs 2.7)

# TODO Check if Python is found!

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/python/src)

set(name pyuvwsim)
add_library(${name} SHARED src/pyuvwsim.c)
target_link_libraries(${name} ${PYTHON_LIBRARIES} uvwsim)
set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX .so)
