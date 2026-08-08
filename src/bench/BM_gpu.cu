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

template <
    typename Solver,
    bool UseMacroCollide,
    bool UseStreamV3,
    bool UseSparseBounce = false,
    bool UseMacroCollideStreamPush = false
>
void AdvanceGPU(LBMFieldGPU& field, dim3 macro_collide_block) {
    if constexpr (UseMacroCollideStreamPush) {
        Solver::macro_collide_stream_push(field);
    } else {
        if constexpr (UseMacroCollide) {
            Solver::macro_collide(field, macro_collide_block);
        } else {
            Solver::macro(field);
            Solver::collide(field);
        }

        if constexpr (UseStreamV3) {
            Solver::stream_v3(field);
        } else {
            Solver::stream(field);
        }
    }

    if constexpr (UseSparseBounce) {
        Solver::bounce_back_sparse(field);
    } else {
        Solver::bounce_back(field);
    }
}

template <
    typename Solver,
    bool UseMacroCollide,
    bool UseStreamV3,
    bool UseSparseBounce = false,
    bool UseMacroCollideStreamPush = false
>
void RunGPUBenchmark(
    benchmark::State& state,
    Layout layout,
    dim3 macro_collide_block = dim3(16, 16)
) {
    const int nx = static_cast<int>(state.range(0));
    const int ny = static_cast<int>(state.range(1));

    constexpr int warmup_iterations = 20;
    constexpr int inner_iterations = 50;

    LBMFieldGPU field(nx, ny, layout);
    Solver::initialize(field);

    // GPU warm-up
    for (int i = 0; i < warmup_iterations; ++i) {
        AdvanceGPU<
            Solver,
            UseMacroCollide,
            UseStreamV3,
            UseSparseBounce,
            UseMacroCollideStreamPush
        >(
            field,
            macro_collide_block
        );
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
            AdvanceGPU<
                Solver,
                UseMacroCollide,
                UseStreamV3,
                UseSparseBounce,
                UseMacroCollideStreamPush
            >(
                field,
                macro_collide_block
            );
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
    RunGPUBenchmark<D2Q9_gpu, false, false>(
        state,
        Layout::AoS
    );
}

static void BM_GPU_Baseline_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, false, false>(
        state,
        Layout::SoA
    );
}

static void BM_GPU_MacroCollide_AoS(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, false>(
        state,
        Layout::AoS
    );
}

static void BM_GPU_Stream_Baseline_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, false>(
        state,
        Layout::SoA
    );
}

static void BM_GPU_Stream_V3_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true>(
        state,
        Layout::SoA
    );
}

static void BM_GPU_SparseBounce_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true, true>(
        state,
        Layout::SoA
    );
}

static void BM_GPU_MacroCollidePushStream_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true, true, true>(
        state,
        Layout::SoA
    );
}

static void BM_GPU_V4_MacroCollide_16x16_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true>(
        state,
        Layout::SoA,
        dim3(16, 16)
    );
}

static void BM_GPU_V4_MacroCollide_32x4_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true>(
        state,
        Layout::SoA,
        dim3(32, 4)
    );
}

static void BM_GPU_V4_MacroCollide_32x8_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true>(
        state,
        Layout::SoA,
        dim3(32, 8)
    );
}

static void BM_GPU_V4_MacroCollide_32x16_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true>(
        state,
        Layout::SoA,
        dim3(32, 16)
    );
}

static void BM_GPU_V4_MacroCollide_64x4_SoA(
    benchmark::State& state
) {
    RunGPUBenchmark<D2Q9_gpu, true, true>(
        state,
        Layout::SoA,
        dim3(64, 4)
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

BENCHMARK(BM_GPU_MacroCollide_AoS)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_Stream_Baseline_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_Stream_V3_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_SparseBounce_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_MacroCollidePushStream_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

#define REGISTER_V4_MACRO_COLLIDE_BENCHMARK(function_name) \
    BENCHMARK(function_name)                               \
        ->Args({512, 256})                                 \
        ->Args({1024, 512})                                \
        ->Args({2048, 1024})                               \
        ->Args({4096, 2048})                               \
        ->UseManualTime()                                  \
        ->Unit(benchmark::kMillisecond)

REGISTER_V4_MACRO_COLLIDE_BENCHMARK(
    BM_GPU_V4_MacroCollide_16x16_SoA
);
REGISTER_V4_MACRO_COLLIDE_BENCHMARK(
    BM_GPU_V4_MacroCollide_32x4_SoA
);
REGISTER_V4_MACRO_COLLIDE_BENCHMARK(
    BM_GPU_V4_MacroCollide_32x8_SoA
);
REGISTER_V4_MACRO_COLLIDE_BENCHMARK(
    BM_GPU_V4_MacroCollide_32x16_SoA
);
REGISTER_V4_MACRO_COLLIDE_BENCHMARK(
    BM_GPU_V4_MacroCollide_64x4_SoA
);

#undef REGISTER_V4_MACRO_COLLIDE_BENCHMARK

BENCHMARK_MAIN();
