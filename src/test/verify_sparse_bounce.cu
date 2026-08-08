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

constexpr double tolerance = 1.0e-5;

struct VerificationResult {
    int nx = 0;
    int ny = 0;
    int steps = 0;
    int boundary_count = 0;
    double max_error = 0.0;
    bool passed = true;
};

void advanceReference(LBMFieldGPU& field) {
    D2Q9_gpu::macro(field);
    D2Q9_gpu::collide(field);
    D2Q9_gpu::stream(field);
    D2Q9_gpu::bounce_back(field);
}

void advanceSparse(LBMFieldGPU& field) {
    D2Q9_gpu::macro(field);
    D2Q9_gpu::collide(field);
    D2Q9_gpu::stream(field);
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

double compareArrays(
    const char* name,
    const std::vector<float>& reference,
    const std::vector<float>& sparse,
    bool& passed
) {
    double max_error = 0.0;
    bool finite = true;

    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(sparse[i])) {
            finite = false;
            max_error = INFINITY;
            break;
        }
        max_error = std::max(
            max_error,
            std::abs(
                static_cast<double>(reference[i])
                - static_cast<double>(sparse[i])
            )
        );
    }

    const bool field_passed = finite && max_error <= tolerance;
    passed &= field_passed;
    std::cout << "  " << name << ": max_error=" << max_error
              << ", " << (field_passed ? "PASS" : "FAIL") << '\n';
    return max_error;
}

VerificationResult runCase(int nx, int ny, int steps) {
    LBMFieldGPU reference(nx, ny, Layout::SoA);
    LBMFieldGPU sparse(nx, ny, Layout::SoA);

    D2Q9_gpu::initialize(reference);
    D2Q9_gpu::initialize(sparse);

    for (int step = 0; step < steps; ++step) {
        advanceReference(reference);
        advanceSparse(sparse);
    }

    // Reconstruct macroscopic fields from the final f state for comparison.
    D2Q9_gpu::macro(reference);
    D2Q9_gpu::macro(sparse);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t cells = static_cast<size_t>(nx) * ny;
    const size_t distributions = cells * D2Q9_gpu::Q;

    VerificationResult result{nx, ny, steps, sparse.boundary_count};
    std::cout << "grid=" << nx << 'x' << ny
              << ", steps=" << steps
              << ", boundary_count=" << sparse.boundary_count << '\n';

    result.max_error = std::max({
        compareArrays(
            "f",
            copyDeviceArray(reference.f, distributions),
            copyDeviceArray(sparse.f, distributions),
            result.passed
        ),
        compareArrays(
            "rho",
            copyDeviceArray(reference.rho, cells),
            copyDeviceArray(sparse.rho, cells),
            result.passed
        ),
        compareArrays(
            "ux",
            copyDeviceArray(reference.ux, cells),
            copyDeviceArray(sparse.ux, cells),
            result.passed
        ),
        compareArrays(
            "uy",
            copyDeviceArray(reference.uy, cells),
            copyDeviceArray(sparse.uy, cells),
            result.passed
        )
    });

    std::cout << "  max_error=" << result.max_error
              << ", result=" << (result.passed ? "PASS" : "FAIL")
              << "\n\n";
    return result;
}

int countBoundaryCells(int nx, int ny) {
    const int center_x = nx / 4;
    const int center_y = ny / 2;
    int count = 0;

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int dx = x - center_x;
            const int dy = y - center_y;
            const bool is_cylinder =
                dx * dx + dy * dy
                <= D2Q9_gpu::cylinder_r * D2Q9_gpu::cylinder_r;
            if (y == 0 || y == ny - 1 || is_cylinder) {
                ++count;
            }
        }
    }
    return count;
}

} // namespace

int main() {
    setGPU();
    std::cout << std::scientific << std::setprecision(8);

    constexpr std::array grids{
        std::array{64, 32},
        std::array{128, 64}
    };
    constexpr std::array steps{1, 10, 100};

    std::vector<VerificationResult> results;
    bool all_passed = true;
    for (const auto& grid : grids) {
        for (const int step_count : steps) {
            VerificationResult result = runCase(
                grid[0],
                grid[1],
                step_count
            );
            all_passed &= result.passed;
            results.push_back(result);
        }
    }

    std::cout << "Grid\tSteps\tBoundary count\tMax error\tResult\n";
    for (const VerificationResult& result : results) {
        std::cout << result.nx << 'x' << result.ny << '\t'
                  << result.steps << '\t'
                  << result.boundary_count << '\t'
                  << result.max_error << '\t'
                  << (result.passed ? "PASS" : "FAIL") << '\n';
    }

    constexpr int statistics_nx = 4096;
    constexpr int statistics_ny = 2048;
    constexpr long long total_cells =
        static_cast<long long>(statistics_nx) * statistics_ny;
    const int boundary_cells = countBoundaryCells(
        statistics_nx,
        statistics_ny
    );
    const double boundary_ratio =
        100.0 * static_cast<double>(boundary_cells)
        / static_cast<double>(total_cells);

    std::cout << "\nBoundary statistics\n"
              << "grid=" << statistics_nx << 'x' << statistics_ny << '\n'
              << "total_cells=" << total_cells << '\n'
              << "boundary_cells=" << boundary_cells << '\n'
              << "boundary_ratio=" << boundary_ratio << "%\n";

    std::cout << "sparse bounce verification: "
              << (all_passed ? "PASS" : "FAIL") << '\n';
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
