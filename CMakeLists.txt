cmake_minimum_required(VERSION 3.7)
project(reed_solomon)

include(CMakeDependentOption)
include(CheckCCompilerFlag)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(CMAKE_C_STANDARD 11)

set(LIB_SOURCE_FILES
        reed_solomon.c
        reed_solomon.h
        reed_solomon_util.c
        reed_solomon_util.h
        field_arithmetic_16.c
        field_arithmetic_16.h
        )

set(BENCH_SOURCE_FILES
        test.c)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

check_c_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    check_c_compiler_flag("/O2" C_FLAG_O2)
    check_c_compiler_flag("/Oy" C_FLAG_Oy)
    check_c_compiler_flag("/Zi" C_FLAG_Zi)
    check_c_compiler_flag("/W4" C_FLAG_W4)

    if(C_FLAG_O2)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2")
    endif()

    if(C_FLAG_Oy)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Oy")
    endif()

    if(C_FLAG_Zi)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Zi")
    endif()

    if(C_FLAG_W4)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "PGI")
    check_c_compiler_flag("-G" C_FLAG_g)
    check_c_compiler_flag("-O0" C_FLAG_O0)
    check_c_compiler_flag("-O3" C_FLAG_O3)
    check_c_compiler_flag("-Minform=inform" C_FLAG_MINFORM_INFORM)

    if(C_FLAG_g)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
    endif(C_FLAG_g)

    if(C_FLAG_O0)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
    endif(C_FLAG_O0)

    if(C_FLAG_O3)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_DEBUG} -O3")
    endif(C_FLAG_O3)

    if(C_FLAG_MINFORM_INFORM)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Minform=inform")
    endif(C_FLAG_MINFORM_INFORM)
else()
    check_c_compiler_flag("-G" C_FLAG_g)
    check_c_compiler_flag("-O0" C_FLAG_O0)
    check_c_compiler_flag("-O3" C_FLAG_O3)
    check_c_compiler_flag("-Wall" C_FLAG_Wall)
    check_c_compiler_flag("-Wextra" C_FLAG_Wextra)

    if(C_FLAG_g)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
    endif(C_FLAG_g)

    if(C_FLAG_O0)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
    endif(C_FLAG_O0)

    if(C_FLAG_O3)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_DEBUG} -O3")
    endif(C_FLAG_O3)

    if(C_FLAG_Wall)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    endif(C_FLAG_Wall)

    if(C_FLAG_Wextra)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
    endif(C_FLAG_Wextra)
endif()

find_package(OpenMP)
cmake_dependent_option(ENABLE_OPENMP "Enable OpenMP support" ON "OPENMP_FOUND" OFF)
if(ENABLE_OPENMP)
    if(NOT OPENMP_FOUND)
        message(FATAL_ERROR "OpenMP support requested, but could not be found")
    endif(NOT OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(ENABLE_OPENMP)

add_library(librsfft STATIC ${LIB_SOURCE_FILES})

add_executable(test_rs ${BENCH_SOURCE_FILES})
target_link_libraries(test_rs librsfft)