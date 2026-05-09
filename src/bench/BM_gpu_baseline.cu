#include <benchmark/benchmark.h>
#include <Layout/lbmField_gpu.hpp>
#include <Layout/layout.hpp>
#include <D2Q9/D2Q9_gpu.cuh>
#include <Utils/Helper/cudaTools.cuh>
#include <chrono>
#include <cuda_profiler_api.h>

template<typename Solver>
void RunGPUBenchmark(benchmark::State& state, Layout layout) {
    int nx = state.range(0);
    int ny = state.range(1);

    LBMFieldGPU field(nx, ny, layout);
    Solver::initialize(field);

    // warm-up
    for (int i = 0; i < 20; ++i) {
        Solver::macro(field);
        Solver::collide(field);
        Solver::stream(field);
        Solver::bounce_back(field);
    }
    cudaDeviceSynchronize();

    const int inner_iters = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        cudaEventRecord(start);

        // Nsight采集 开始
        cudaProfilerStart();

        for (int i = 0; i < inner_iters; ++i) {
            Solver::macro(field);
            Solver::collide(field);
            Solver::stream(field);
            Solver::bounce_back(field);
        }

        // Nsight采集 结束
        cudaProfilerStop();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double seconds = milliseconds / 1000.0;

        state.SetIterationTime(seconds);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double updates = static_cast<double>(nx) * ny * inner_iters * state.iterations();
    state.counters["MLUPS"] = benchmark::Counter(
        updates,
        benchmark::Counter::kIsRate,
        benchmark::Counter::kIs1000
    );
}

static void BM_GPU_Baseline_AoS(benchmark::State& state) {
    RunGPUBenchmark<D2Q9_gpu>(state, Layout::AoS);
}

static void BM_GPU_Baseline_SoA(benchmark::State& state) {
    RunGPUBenchmark<D2Q9_gpu>(state, Layout::SoA);
}

// BENCHMARK(BM_GPU_Baseline_AoS)
//     ->Args({512, 256})
//     ->Args({1024, 512})
//     ->Args({2048, 1024})
//     ->Args({4096, 2048})
//     ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_Baseline_SoA)
    // ->Args({512, 256})
    // ->Args({1024, 512})
    // ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();