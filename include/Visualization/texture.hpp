#pragma once
#include <GL/gl.h>
#include <Layout/lbmField_gpu.hpp>
#include <Layout/lbmField_cpu.hpp>
#include <cuda_runtime.h>
#include <vector>

inline void fetchVelocity(const LBMFieldCPU& field, std::vector<float>& ux, std::vector<float>& uy) {
    ux = field.ux;
    uy = field.uy;
}

inline void fetchVelocity(const LBMFieldGPU& field, std::vector<float>& ux, std::vector<float>& uy) {
    int cells = field.nx * field.ny;
    ux.resize(cells);
    uy.resize(cells);

    cudaMemcpy(ux.data(), field.ux, cells * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(uy.data(), field.uy, cells * sizeof(float), cudaMemcpyDeviceToHost);
}

inline void fetchDensity(const LBMFieldCPU& field, std::vector<float>& rho) {
    rho = field.rho;
}

inline void fetchDensity(const LBMFieldGPU& field, std::vector<float>& rho) {
    const int cells = field.nx * field.ny;
    rho.resize(cells);
    CUDA_CHECK(cudaMemcpy(rho.data(), field.rho, cells * sizeof(float), cudaMemcpyDeviceToHost));
}

__global__ void buildTextureKernel(const float* ux, const float* uy, float* tex, int nx, int ny);

inline void updateTextureGPU(LBMFieldGPU& field, GLuint texID) {
    int threads = 256;
    int blocks = (field.nx * field.ny + threads - 1) / threads;

    buildTextureKernel<<<blocks, threads>>>(field.ux, field.uy, field.d_tex, field.nx, field.ny);
    cudaDeviceSynchronize();

    std::vector<float> texCPU(field.nx * field.ny);
    CUDA_CHECK(cudaMemcpy(texCPU.data(), field.d_tex, field.nx * field.ny * sizeof(float), cudaMemcpyDeviceToHost));
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, field.nx, field.ny, GL_RED, GL_FLOAT, texCPU.data());
}

inline GLuint initTexture(int nx, int ny) {
    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    return texID;
}
