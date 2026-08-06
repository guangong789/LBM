#include <benchmark/benchmark.h>
#include <Layout/lbmField_cpu.hpp>
#include <Layout/layout.hpp>
#include <D2Q9/D2Q9_omp.hpp>
#include <omp.h>

template<typename Solver>
static void RunOpenMPBenchmark(benchmark::State& state, Layout layout) {
    int nx = state.range(0);
    int ny = state.range(1);
    int num_threads = state.range(2);

    omp_set_num_threads(num_threads);

    LBMFieldCPU field(nx, ny, layout);
    Solver::initialize(field);

    // warm-up
    for (int i = 0; i < 10; ++i) {
        Solver::macro(field);
        Solver::collide(field);
        Solver::stream(field);
        Solver::bounce_back(field);
    }

    for (auto _ : state) {
        Solver::macro(field);
        Solver::collide(field);
        Solver::stream(field);
        Solver::bounce_back(field);

        benchmark::DoNotOptimize(field.f.data());
    }

    double updates_per_iter = static_cast<double>(nx) * ny;

    state.counters["MLUPS"] =
        benchmark::Counter(
            updates_per_iter,
            benchmark::Counter::kIsIterationInvariantRate,
            benchmark::Counter::OneK::kIs1000
        );

    state.counters["Threads"] = num_threads;
}

static void BM_OMP_AoS(benchmark::State& state) {
    RunOpenMPBenchmark<D2Q9_cpu_omp>(state, Layout::AoS);
}

#define REGISTER_GRID(nx, ny) \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 1}); \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 2}); \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 4}); \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 8}); \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 10}); \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 16}); \
    BENCHMARK(BM_OMP_AoS)->Args({nx, ny, 20});

REGISTER_GRID(512, 256);
REGISTER_GRID(1024, 512);
REGISTER_GRID(2048, 1024);
REGISTER_GRID(4096, 2048);

BENCHMARK_MAIN();