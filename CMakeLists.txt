cmake_minimum_required(VERSION 3.20)
project(cuda_inference_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture
# sm_89 = Ada Lovelace (RTX 4050/4060/4070/4080)
# RTX 4050 Laptop GPU — Ada Lovelace, 6GB GDDR6
set(CMAKE_CUDA_ARCHITECTURES 89)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# CUDA compile flags
set(CUDA_NVCC_FLAGS
    --use_fast_math          # fused multiply-add, fast intrinsics
    --expt-relaxed-constexpr
    -lineinfo                # for Nsight profiling (remove in prod)
    -O3
)

# Kernel objects
add_library(kernels OBJECT
    kernels/gemm.cu
    kernels/activation.cu
    kernels/norm.cu
)
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# Python module (pybind11)
pybind11_add_module(cuda_engine
    python/binding.cpp
    $<TARGET_OBJECTS:kernels>
)
target_link_libraries(cuda_engine PRIVATE
    CUDA::cudart
    CUDA::cublas    # only for benchmark comparison, not core kernels
)
target_include_directories(cuda_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)

# Build info
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")