
# Flag to set verbose compiler and build information when running cmake
set(BUILD_VERBOSE ON)

# Set default install paths
if (NOT UVWSIM_LIB_INSTALL_DIR)
    set(UVWSIM_LIB_INSTALL_DIR "lib")
endif()
if (NOT UVWSIM_INCLUDE_INSTALL_DIR)
    set(UVWSIM_INCLUDE_INSTALL_DIR "include")
endif()

# If not defined, build a static library.
if (NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
endif()

# If not set, build in release mode.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

if (NOT WIN32)
    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "-fPIC")
        set(CMAKE_C_FLAGS "-fPIC")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wno-unused-function")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wno-unused-function")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
    set(CMAKE_C_FLAGS_MINSIZEREL "-O1 -DNDEBUG")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # using Clang or GNU compilers
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-show-option")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-align")
        #set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-qual")
        # Warning suppressions for the Python interface (TODO: Fix these...).
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-parameter")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-label")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-long-long")


        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-show-option")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pedantic")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcast-align")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcast-qual")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-long-long")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-variadic-macros")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-missing-field-initializers")

    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # using Intel compilers
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fp-model precise")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wremarks")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcheck")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -wd2259")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -wd981")

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcheck")
    endif()
else()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # MSVC compiler - This is currently a placeholder.
    endif()
endif()


# Print some information on the build options.
if (BUILD_VERBOSE)
    message(STATUS "")
    message(STATUS "***********************************************************")
    message(STATUS "Build type   : ${CMAKE_BUILD_TYPE}")
    if (BUILD_SHARED_LIBS MATCHES ON)
        message(STATUS "Library type : shared")
    else()
        message(STATUS "Library type : static")
    endif()
    message(STATUS "C++ compiler : ${CMAKE_CXX_COMPILER}")
    message(STATUS "C compiler   : ${CMAKE_C_COMPILER}")
    message(STATUS "Compiler ID  : ${CMAKE_CXX_COMPILER_ID}")
    if (${CMAKE_BUILD_TYPE} MATCHES release)
        message(STATUS "C++ flags    : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
        message(STATUS "C flags      : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES debug)
        message(STATUS "C++ flags    : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "C flags      : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES relwithdebinfo)
        message(STATUS "C++ flags    : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        message(STATUS "C flags      : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES minsizerel)
        message(STATUS "C++ flags    : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
        message(STATUS "C flags      : ${CMAKE_C_FLAGS}$ {CMAKE_C_FLAGS_MINSIZEREL}")
    endif()
    message(STATUS "***********************************************************")
    message(STATUS "")
endif(BUILD_VERBOSE)
