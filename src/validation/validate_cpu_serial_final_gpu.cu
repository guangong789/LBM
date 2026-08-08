#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <D2Q9/D2Q9_cpu.hpp>
#include <D2Q9/D2Q9_gpu.cuh>
#include <Layout/layout.hpp>
#include <Layout/lbmField_cpu.hpp>
#include <Layout/lbmField_gpu.hpp>
#include <Utils/Helper/cudaTools.cuh>

namespace {

struct ErrorMetrics {
    double l2;
    float max;
};

ErrorMetrics compareField(
    const std::vector<float>& cpu,
    const std::vector<float>& gpu
) {
    assert(cpu.size() == gpu.size());

    float max_error = 0.0F;
    double squared_error_sum = 0.0;

    for (size_t i = 0; i < cpu.size(); ++i) {
        const float error = std::abs(cpu[i] - gpu[i]);
        max_error = std::max(max_error, error);
        squared_error_sum += static_cast<double>(error) * error;
    }

    return {
        std::sqrt(squared_error_sum / static_cast<double>(cpu.size())),
        max_error
    };
}

std::vector<float> copyDeviceField(const float* device, size_t count) {
    std::vector<float> host(count);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device,
        count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    return host;
}

void printValidationErrors(
    const LBMFieldCPU& cpu,
    const LBMFieldGPU& gpu
) {
    const size_t cells = static_cast<size_t>(cpu.nx) * cpu.ny;
    const std::vector<float> rho_gpu = copyDeviceField(gpu.rho, cells);
    const std::vector<float> ux_gpu = copyDeviceField(gpu.ux, cells);
    const std::vector<float> uy_gpu = copyDeviceField(gpu.uy, cells);

    const ErrorMetrics rho_error = compareField(cpu.rho, rho_gpu);
    const ErrorMetrics ux_error = compareField(cpu.ux, ux_gpu);
    const ErrorMetrics uy_error = compareField(cpu.uy, uy_gpu);

    std::cout << std::scientific
              << "rho: L2 = " << rho_error.l2
              << ", Max = " << rho_error.max
              << " | ux: L2 = " << ux_error.l2
              << ", Max = " << ux_error.max
              << " | uy: L2 = " << uy_error.l2
              << ", Max = " << uy_error.max
              << std::endl;
}

void advanceCpuSerial(LBMFieldCPU& field) {
    D2Q9_cpu::macro(field);
    D2Q9_cpu::collide(field);
    D2Q9_cpu::stream(field);
    D2Q9_cpu::bounce_back(field);
}

void advanceFinalGpu(LBMFieldGPU& field) {
    D2Q9_gpu::macro_collide_stream_push(field);
    D2Q9_gpu::bounce_back_sparse(field);
}

} // namespace

int main() {
    constexpr int nx = 2048;
    constexpr int ny = 1024;
    constexpr int total_steps = 8000;
    constexpr int record_interval = 100;

    std::cout << "Running validation (CPU Serial vs Final Fused GPU)"
              << std::endl;

    setGPU();
    LBMFieldCPU cpu(nx, ny, Layout::SoA);
    LBMFieldGPU gpu(nx, ny, Layout::SoA);

    D2Q9_cpu::initialize(cpu);
    D2Q9_gpu::initialize(gpu);

    for (int step = 0; step < total_steps; ++step) {
        advanceCpuSerial(cpu);
        advanceFinalGpu(gpu);

        if (step % record_interval == 0) {
            // Reconstruct macros from the same completed distribution state.
            D2Q9_cpu::macro(cpu);
            D2Q9_gpu::macro(gpu);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::cout << "[Step " << step << "] ";
            printValidationErrors(cpu, gpu);
        }
    }

    std::cout << "Validation finished." << std::endl;
    return 0;
}
