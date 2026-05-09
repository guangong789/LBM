cmake_minimum_required(VERSION 4.0)
project(LBM LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(COMMON_INCLUDES
    ${PROJECT_SOURCE_DIR}/include      
    ${PROJECT_SOURCE_DIR}/glm         
    ${PROJECT_SOURCE_DIR}/glad/include
    /usr/local/cuda/include
)

# CUDA flags
set(COMMON_CUDA_FLAGS
    --expt-relaxed-constexpr
    --use_fast_math
    -O3
)

# OpenMP 支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallel CPU solver")
    set(USE_OPENMP TRUE)
else()
    message(STATUS "OpenMP not found, using serial CPU only")
    set(USE_OPENMP FALSE)
endif()

find_package(OpenGL REQUIRED)

add_library(glad STATIC glad/src/glad.c)
target_include_directories(glad PUBLIC glad/include)

add_subdirectory(glfw)

# solver
add_library(lbm_core STATIC
    src/D2Q9/D2Q9_cpu.cpp
    src/utility.cpp
)

# OpenMP 版本
if(USE_OPENMP)
    target_sources(lbm_core PRIVATE src/D2Q9/D2Q9_omp.cpp)
endif()

# GPU 版本
target_sources(lbm_core PRIVATE src/D2Q9/D2Q9_gpu.cu)

target_include_directories(lbm_core PUBLIC ${COMMON_INCLUDES})

# 编译选项
target_compile_options(lbm_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# OpenMP 链接
if(USE_OPENMP)
    target_link_libraries(lbm_core OpenMP::OpenMP_CXX)
    target_compile_definitions(lbm_core PRIVATE _OPENMP)
endif()

set_target_properties(lbm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# main
add_executable(main
    src/main.cu
    src/shader.cpp
    src/mesh.cpp
    src/texture.cu
)

target_include_directories(main PRIVATE ${COMMON_INCLUDES})

target_link_libraries(main
    lbm_core
    glad
    glfw
    ${OPENGL_gl_LIBRARY}
    dl
    pthread
)

# OpenMP 链接到 main
if(USE_OPENMP)
    target_link_libraries(main OpenMP::OpenMP_CXX)
endif()

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# benchmark
find_package(benchmark REQUIRED)

# benchmark - CPU serial version (AoS vs SoA)
add_executable(benchmark_serial
    src/bench/BM_serial.cpp
)

target_include_directories(benchmark_serial PRIVATE ${COMMON_INCLUDES})

target_link_libraries(benchmark_serial
    lbm_core
    benchmark::benchmark
    pthread
)

target_compile_options(benchmark_serial PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# benchmark - CPU parallel (OpenMP) version
add_executable(benchmark_omp
    src/bench/BM_omp.cpp
)

target_include_directories(benchmark_omp PRIVATE ${COMMON_INCLUDES})

target_link_libraries(benchmark_omp
    lbm_core
    benchmark::benchmark
    pthread
)

if(USE_OPENMP)
    target_link_libraries(benchmark_omp OpenMP::OpenMP_CXX)
endif()

target_compile_options(benchmark_omp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# benchmark - GPU baseline version (AoS vs SoA)
add_executable(benchmark_gpu_baseline
    src/bench/BM_gpu_baseline.cu
)

target_include_directories(benchmark_gpu_baseline PRIVATE ${COMMON_INCLUDES})

target_link_libraries(benchmark_gpu_baseline
    lbm_core
    benchmark::benchmark
    pthread
)

set_target_properties(benchmark_gpu_baseline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(benchmark_gpu_baseline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)