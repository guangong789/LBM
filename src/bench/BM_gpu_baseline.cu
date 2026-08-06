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
        Solver::macro(field);
        Solver::collide(field);
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
            Solver::macro(field);
            Solver::collide(field);
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

static void BM_GPU_Baseline_AoS(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu>(
        state,
        Layout::AoS
    );
}

static void BM_GPU_Baseline_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu>(
        state,
        Layout::SoA
    );
}

}  // namespace

BENCHMARK(BM_GPU_Baseline_AoS)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_Baseline_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();