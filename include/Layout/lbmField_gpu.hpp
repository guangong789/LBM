#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <Layout/layout.hpp>
#include <Layout/lbmFieldView.cuh>
#include <Utils/Helper/cudaTools.cuh>

struct LBMFieldGPU {
    float* f = nullptr;
    float* f_next = nullptr;
    float* rho = nullptr;
    float* ux = nullptr;
    float* uy = nullptr;
    unsigned char* is_cylinder = nullptr;

    int nx = 0, ny = 0;
    Layout layout;

    // GPU纹理缓冲
    float* d_tex = nullptr;

    LBMFieldGPU(int nx_, int ny_, Layout layout_) : nx(nx_), ny(ny_), layout(layout_) {
        int cells = nx * ny;

        CUDA_CHECK(cudaMalloc(&f, cells * 9 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&f_next, cells * 9 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&rho, cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ux,  cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&uy,  cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&is_cylinder, cells * sizeof(unsigned char)));

        CUDA_CHECK(cudaMemset(f, 0, cells * 9 * sizeof(float)));
        CUDA_CHECK(cudaMemset(f_next, 0, cells * 9 * sizeof(float)));
        CUDA_CHECK(cudaMemset(rho, 0, cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(ux,  0, cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(uy,  0, cells * sizeof(float)));
        CUDA_CHECK(cudaMemset(is_cylinder, 0, cells * sizeof(unsigned char)));

        // 分配纹理缓冲
        CUDA_CHECK(cudaMalloc(&d_tex, cells * sizeof(float)));
    }

    ~LBMFieldGPU() {
        cudaFree(f);
        cudaFree(f_next);
        cudaFree(rho);
        cudaFree(ux);
        cudaFree(uy);
        cudaFree(is_cylinder);
        cudaFree(d_tex);
    }

    LBMFieldView view() const {
        return LBMFieldView(f, f_next, rho, ux, uy, is_cylinder, nx, ny, layout);
    }
};