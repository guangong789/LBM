#include <benchmark/benchmark.h>

#include <Layout/lbmField_gpu.hpp>
#include <Layout/layout.hpp>
#include <D2Q9/D2Q9_gpu.cuh>
#include <Utils/Helper/cudaTools.cuh>

#include <cuda_runtime.h>

namespace {

inline void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            cudaGetErrorString(result)
        );
    }
}

__device__ __constant__ int benchmark_cx[9] = {
    0, 1, 0, -1, 0, 1, -1, -1, 1
};

__device__ __constant__ int benchmark_cy[9] = {
    0, 0, 1, 0, -1, 1, 1, -1, -1
};

__device__ __constant__ float benchmark_w[9] = {
    4.f / 9.f,
    1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
    1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f
};

__global__ void macro_collide_kernel(LBMFieldView v) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    const int cell = y * v.nx + x;
    const int cells = v.nx * v.ny;
    float distributions[9];

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int id = (v.layout == Layout::AoS)
            ? cell * 9 + k
            : k * cells + cell;
        distributions[k] = v.f[id];
    }

    // Preserve the baseline collision behavior inside solid cylinder cells.
    if (v.is_cylinder[cell]) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int id = (v.layout == Layout::AoS)
                ? cell * 9 + k
                : k * cells + cell;
            v.f_next[id] = distributions[k];
        }
        return;
    }

    float rho = 0.f;
    float momentum_x = 0.f;
    float momentum_y = 0.f;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        rho += distributions[k];
        momentum_x += distributions[k] * benchmark_cx[k];
        momentum_y += distributions[k] * benchmark_cy[k];
    }

    if (rho < 1e-6f) rho = 1e-6f;

    const float ux = momentum_x / rho;
    const float uy = momentum_y / rho;
    const float u2 = ux * ux + uy * uy;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int id = (v.layout == Layout::AoS)
            ? cell * 9 + k
            : k * cells + cell;
        const float cu = 3.f * (
            ux * benchmark_cx[k] + uy * benchmark_cy[k]
        );
        const float feq = benchmark_w[k] * rho * (
            1.f + cu + 0.5f * cu * cu - 1.5f * u2
        );
        float updated = distributions[k]
            - (distributions[k] - feq) / D2Q9_gpu::tau;

        if (isnan(updated)) updated = feq;
        v.f_next[id] = updated;
    }
}

void MacroCollide(LBMFieldGPU& field) {
    dim3 block(16, 16);
    dim3 grid(
        (field.nx + 15) / 16,
        (field.ny + 15) / 16
    );
    macro_collide_kernel<<<grid, block>>>(field.view());
    CheckCuda(cudaGetLastError(), "macro_collide_kernel launch");
}

template <typename Solver>
void RunGPUBenchmark(
    benchmark::State& state,
    Layout layout
) {
    const int nx = static_cast<int>(state.range(0));
    const int ny = static_cast<int>(state.range(1));

    constexpr int warmup_iterations = 20;
    constexpr int inner_iterations = 50;

    LBMFieldGPU field(nx, ny, layout);
    Solver::initialize(field);

    // GPU warm-up
    for (int i = 0; i < warmup_iterations; ++i) {
        MacroCollide(field);
        Solver::stream(field);
        Solver::bounce_back(field);
    }

    CheckCuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize after warm-up"
    );

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CheckCuda(
        cudaEventCreate(&start),
        "cudaEventCreate(start)"
    );

    CheckCuda(
        cudaEventCreate(&stop),
        "cudaEventCreate(stop)"
    );

    for (auto _ : state) {
        CheckCuda(
            cudaEventRecord(start),
            "cudaEventRecord(start)"
        );

        for (int i = 0; i < inner_iterations; ++i) {
            MacroCollide(field);
            Solver::stream(field);
            Solver::bounce_back(field);
        }

        CheckCuda(
            cudaEventRecord(stop),
            "cudaEventRecord(stop)"
        );

        CheckCuda(
            cudaEventSynchronize(stop),
            "cudaEventSynchronize(stop)"
        );

        float elapsed_milliseconds = 0.0F;

        CheckCuda(
            cudaEventElapsedTime(
                &elapsed_milliseconds,
                start,
                stop
            ),
            "cudaEventElapsedTime"
        );

        const double elapsed_seconds =
            static_cast<double>(elapsed_milliseconds) / 1000.0;

        state.SetIterationTime(elapsed_seconds);
    }

    CheckCuda(
        cudaEventDestroy(start),
        "cudaEventDestroy(start)"
    );

    CheckCuda(
        cudaEventDestroy(stop),
        "cudaEventDestroy(stop)"
    );

    /*
     * 每次 Google Benchmark 迭代执行 inner_iterations 个时间步，
     * 每个时间步更新 nx × ny 个格点。
     */
    const double updates_per_benchmark_iteration =
        static_cast<double>(nx) *
        static_cast<double>(ny) *
        static_cast<double>(inner_iterations);

    state.counters["MLUPS"] = benchmark::Counter(
        updates_per_benchmark_iteration,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );

    state.counters["InnerIterations"] =
        static_cast<double>(inner_iterations);
}

static void BM_GPU_MacroCollide_AoS(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu>(
        state,
        Layout::AoS
    );
}

static void BM_GPU_MacroCollide_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu>(
        state,
        Layout::SoA
    );
}

}  // namespace

BENCHMARK(BM_GPU_MacroCollide_AoS)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_MacroCollide_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
