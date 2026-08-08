#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <D2Q9/D2Q9_gpu.cuh>
#include <Layout/lbmField_gpu.hpp>
#include <Layout/layout.hpp>
#include <Utils/Helper/cudaTools.cuh>

namespace {

constexpr double tolerance = 0.0;

struct TestCase {
    int nx;
    int ny;
    int steps;
};

struct VerificationResult {
    int nx = 0;
    int ny = 0;
    int steps = 0;
    double f_error = 0.0;
    double rho_error = 0.0;
    double ux_error = 0.0;
    double uy_error = 0.0;
    bool passed = true;
};

void advanceReference(LBMFieldGPU& field) {
    D2Q9_gpu::macro_collide(field);
    D2Q9_gpu::stream_v3(field);
    D2Q9_gpu::bounce_back_sparse(field);
}

void advanceCandidate(LBMFieldGPU& field) {
    D2Q9_gpu::macro_collide_stream_push(field);
    D2Q9_gpu::bounce_back_sparse(field);
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

double calculateMaxError(
    const std::vector<float>& reference,
    const std::vector<float>& candidate
) {
    double max_error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            return INFINITY;
        }
        max_error = std::max(
            max_error,
            std::abs(
                static_cast<double>(reference[i])
                - static_cast<double>(candidate[i])
            )
        );
    }
    return max_error;
}

VerificationResult runCase(int nx, int ny, int steps) {
    LBMFieldGPU reference(nx, ny, Layout::SoA);
    LBMFieldGPU candidate(nx, ny, Layout::SoA);

    D2Q9_gpu::initialize(reference);
    D2Q9_gpu::initialize(candidate);

    for (int step = 0; step < steps; ++step) {
        advanceReference(reference);
        advanceCandidate(candidate);
    }

    // The optimized timestep keeps macroscopic values in registers. Rebuild
    // both fields from their final distributions solely for verification.
    D2Q9_gpu::macro(reference);
    D2Q9_gpu::macro(candidate);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t cells = static_cast<size_t>(nx) * ny;
    const size_t distribution_count = cells * D2Q9_gpu::Q;

    VerificationResult result;
    result.nx = nx;
    result.ny = ny;
    result.steps = steps;
    result.f_error = calculateMaxError(
        copyDeviceArray(reference.f, distribution_count),
        copyDeviceArray(candidate.f, distribution_count)
    );
    result.rho_error = calculateMaxError(
        copyDeviceArray(reference.rho, cells),
        copyDeviceArray(candidate.rho, cells)
    );
    result.ux_error = calculateMaxError(
        copyDeviceArray(reference.ux, cells),
        copyDeviceArray(candidate.ux, cells)
    );
    result.uy_error = calculateMaxError(
        copyDeviceArray(reference.uy, cells),
        copyDeviceArray(candidate.uy, cells)
    );

    result.passed = result.f_error <= tolerance
        && result.rho_error <= tolerance
        && result.ux_error <= tolerance
        && result.uy_error <= tolerance;

    std::cout << "grid=" << nx << 'x' << ny << '\n'
              << "steps=" << steps << '\n'
              << "max f error=" << result.f_error << '\n'
              << "max rho error=" << result.rho_error << '\n'
              << "max ux error=" << result.ux_error << '\n'
              << "max uy error=" << result.uy_error << '\n'
              << "result=" << (result.passed ? "PASS" : "FAIL")
              << "\n\n";
    return result;
}

} // namespace

int main() {
    setGPU();
    std::cout << std::scientific << std::setprecision(8);

    constexpr std::array cases{
        TestCase{64, 32, 1},
        TestCase{64, 32, 10},
        TestCase{64, 32, 100},
        TestCase{128, 64, 1},
        TestCase{128, 64, 10},
        TestCase{128, 64, 100},
        TestCase{65, 33, 1},
        TestCase{65, 33, 10},
        TestCase{65, 33, 100},
        TestCase{127, 65, 1},
        TestCase{127, 65, 10},
        TestCase{127, 65, 100},
        TestCase{257, 129, 500},
        TestCase{511, 257, 1000}
    };

    std::vector<VerificationResult> results;
    bool all_passed = true;
    for (const TestCase& test_case : cases) {
        VerificationResult result = runCase(
            test_case.nx,
            test_case.ny,
            test_case.steps
        );
        all_passed &= result.passed;
        results.push_back(result);
    }

    std::cout << "Grid\tSteps\tMax Error\tResult\n";
    for (const VerificationResult& result : results) {
        const double max_error = std::max({
            result.f_error,
            result.rho_error,
            result.ux_error,
            result.uy_error
        });
        std::cout << result.nx << 'x' << result.ny << '\t'
                  << result.steps << '\t'
                  << max_error << '\t'
                  << (result.passed ? "PASS" : "FAIL") << '\n';
    }

    std::cout << "macro-collide push-stream verification: "
              << (all_passed ? "PASS" : "FAIL") << '\n';
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
