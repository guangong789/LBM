#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <D2Q9/D2Q9_gpu.cuh>
#include <Layout/lbmField_gpu.hpp>
#include <Layout/layout.hpp>
#include <Utils/Helper/cudaTools.cuh>

namespace {

constexpr double tolerance = 1.0e-5;

struct ErrorMetrics {
    double max_absolute = 0.0;
    double l2 = 0.0;
    double mean_absolute = 0.0;
    bool finite = true;
};

struct VerificationResult {
    bool passed = true;
    double max_error = 0.0;
};

struct BlockConfiguration {
    const char* name;
    unsigned int x;
    unsigned int y;
};

void advanceReference(LBMFieldGPU& field) {
    D2Q9_gpu::macro(field);
    D2Q9_gpu::collide(field);
    D2Q9_gpu::stream(field);
    D2Q9_gpu::bounce_back(field);
}

void advanceOptimized(LBMFieldGPU& field, dim3 macro_collide_block) {
    D2Q9_gpu::macro_collide(field, macro_collide_block);
    D2Q9_gpu::stream_v3(field);
    D2Q9_gpu::bounce_back(field);
}

std::vector<float> copyDeviceArray(const float* device, size_t count) {
    std::vector<float> host(count);
    CUDA_CHECK(cudaMemcpy(
        host.data(),
        device,
        count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    return host;
}

ErrorMetrics calculateError(
    const std::vector<float>& reference,
    const std::vector<float>& optimized
) {
    ErrorMetrics metrics;
    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(optimized[i])) {
            metrics.finite = false;
            continue;
        }

        const double error = std::abs(
            static_cast<double>(reference[i])
            - static_cast<double>(optimized[i])
        );
        metrics.max_absolute = std::max(metrics.max_absolute, error);
        squared_error_sum += error * error;
        absolute_error_sum += error;
    }

    if (!metrics.finite) {
        metrics.max_absolute = INFINITY;
        metrics.l2 = INFINITY;
        metrics.mean_absolute = INFINITY;
        return metrics;
    }

    const double count = static_cast<double>(reference.size());
    metrics.l2 = std::sqrt(squared_error_sum / count);
    metrics.mean_absolute = absolute_error_sum / count;
    return metrics;
}

ErrorMetrics reportField(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& optimized
) {
    const ErrorMetrics metrics = calculateError(reference, optimized);
    const bool passed = metrics.finite
        && metrics.max_absolute <= tolerance;

    std::cout << "  " << name
              << ": max_abs=" << metrics.max_absolute
              << ", l2=" << metrics.l2
              << ", mean_abs=" << metrics.mean_absolute
              << ", " << (passed ? "PASS" : "FAIL") << '\n';
    return metrics;
}

VerificationResult runCase(
    int nx,
    int ny,
    int timesteps,
    const BlockConfiguration& configuration
) {
    LBMFieldGPU reference(nx, ny, Layout::SoA);
    LBMFieldGPU optimized(nx, ny, Layout::SoA);

    D2Q9_gpu::initialize(reference);
    D2Q9_gpu::initialize(optimized);

    for (int step = 0; step < timesteps; ++step) {
        advanceReference(reference);
        advanceOptimized(
            optimized,
            dim3(configuration.x, configuration.y)
        );
    }

    // Reconstruct macroscopic fields from the final distribution state. This
    // is outside the optimized timestep and does not alter f.
    D2Q9_gpu::macro(reference);
    D2Q9_gpu::macro(optimized);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t cells = static_cast<size_t>(nx) * ny;
    const size_t distributions = cells * D2Q9_gpu::Q;

    const std::vector<float> f_reference = copyDeviceArray(
        reference.f,
        distributions
    );
    const std::vector<float> f_optimized = copyDeviceArray(
        optimized.f,
        distributions
    );
    const std::vector<float> rho_reference = copyDeviceArray(
        reference.rho,
        cells
    );
    const std::vector<float> rho_optimized = copyDeviceArray(
        optimized.rho,
        cells
    );
    const std::vector<float> ux_reference = copyDeviceArray(
        reference.ux,
        cells
    );
    const std::vector<float> ux_optimized = copyDeviceArray(
        optimized.ux,
        cells
    );
    const std::vector<float> uy_reference = copyDeviceArray(
        reference.uy,
        cells
    );
    const std::vector<float> uy_optimized = copyDeviceArray(
        optimized.uy,
        cells
    );

    std::cout << "configuration=" << configuration.name
              << ", grid=" << nx << 'x' << ny
              << ", timesteps=" << timesteps
              << ", tolerance=" << tolerance << '\n';

    VerificationResult result;
    for (const ErrorMetrics metrics : {
             reportField("f", f_reference, f_optimized),
             reportField("rho", rho_reference, rho_optimized),
             reportField("ux", ux_reference, ux_optimized),
             reportField("uy", uy_reference, uy_optimized)
         }) {
        result.passed &= metrics.finite
            && metrics.max_absolute <= tolerance;
        result.max_error = std::max(
            result.max_error,
            metrics.max_absolute
        );
    }

    std::cout << "  max_error=" << result.max_error
              << ", result=" << (result.passed ? "PASS" : "FAIL")
              << "\n\n";
    return result;
}

} // namespace

int main() {
    setGPU();
    std::cout << std::scientific << std::setprecision(8);

    constexpr std::array configurations{
        BlockConfiguration{"16x16", 16, 16},
        BlockConfiguration{"32x4", 32, 4},
        BlockConfiguration{"32x8", 32, 8},
        BlockConfiguration{"32x16", 32, 16},
        BlockConfiguration{"64x4", 64, 4}
    };

    bool all_passed = true;
    std::array<VerificationResult, configurations.size()> results{};

    for (size_t i = 0; i < configurations.size(); ++i) {
        results[i] = runCase(128, 64, 100, configurations[i]);
        all_passed &= results[i].passed;
    }

    std::cout << "Block\tResult\tMax Error\n";
    for (size_t i = 0; i < configurations.size(); ++i) {
        std::cout << configurations[i].name << '\t'
                  << (results[i].passed ? "PASS" : "FAIL") << '\t'
                  << results[i].max_error << '\n';
    }

    std::cout << "optimized GPU verification: "
              << (all_passed ? "PASS" : "FAIL") << '\n';
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
