#pragma once
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <Utils/Helper/helper_cuda.h>

#define CUDA_CHECK(x) \
{ \
    cudaError_t err = x; \
    if(err != cudaSuccess){ \
        std::cout << "CUDA ERROR: " \
                  << cudaGetErrorString(err) \
                  << std::endl; \
        exit(1); \
    } \
}

#define CUDA_CHECK_KERNEL() \
{ \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) \
        std::cout << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl; \
}

inline int getSPcores(const cudaDeviceProp& prop) {
    int core = 0;
    int mp = prop.multiProcessorCount;
    switch (prop.major) {
        case 2: core = (prop.minor == 1) ? mp * 48 : mp * 32; break;
        case 3: core = mp * 192; break;
        case 5: core = mp * 128; break;
        case 6:
            if (prop.minor == 0) core = mp * 64;
            else if (prop.minor == 1 || prop.minor == 2) core = mp * 128;
            break;
        case 7: core = mp * 64; break;
        case 8:
            if (prop.minor == 0) core = mp * 64;
            else if (prop.minor == 6 || prop.minor == 9) core = mp * 128;
            break;
        case 9: core = mp * 128; break;
        default: printf("Unknown device type (SM %d.%d)\n", prop.major, prop.minor); break;
    }
    return core;
}

inline void setGPU() {
    int count = 0;
    checkCudaErrors(cudaGetDeviceCount(&count));
    if (count == 0) {
        printf("No CUDA compatible GPU found\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    checkCudaErrors(cudaSetDevice(dev));

    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
}