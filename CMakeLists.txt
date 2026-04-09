cmake_minimum_required(VERSION 3.18)

# Set CUDA architecture before project() command
# Default: sm_80 (A100). Override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(GTSPP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
aux_source_directory(src SRC_LIST)
include_directories(include)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math>
)
