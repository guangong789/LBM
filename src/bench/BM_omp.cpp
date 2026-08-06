#include <benchmark/benchmark.h>
#include <Layout/lbmField_cpu.hpp>
#include <Layout/layout.hpp>
#include <D2Q9/D2Q9_omp.hpp>
#include <omp.h>

template<typename Solver>
static void RunOpenMPBenchmark(
    benchmark::State& state,
    Layout layout
) {
    const int nx = static_cast<int>(state.range(0));
    const int ny = static_cast<int>(state.range(1));
    const int num_threads = static_cast<int>(state.range(2));

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    LBMFieldCPU field(nx, ny, layout);
    Solver::initialize(field);

    // Warm-up
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
        benchmark::ClobberMemory();
    }

    const double updates_per_iteration =
        static_cast<double>(nx) * static_cast<double>(ny);

    state.counters["MLUPS"] = benchmark::Counter(
        updates_per_iteration,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );

    state.counters["Threads"] =
        static_cast<double>(num_threads);
}

static void BM_OMP_AoS(benchmark::State& state) {
    RunOpenMPBenchmark<D2Q9_cpu_omp>(
        state,
        Layout::AoS
    );
}

#define REGISTER_OMP_CASE(nx, ny, threads)               \
    BENCHMARK(BM_OMP_AoS)                                \
        ->Args({nx, ny, threads})                        \
        ->UseRealTime()

#define REGISTER_GRID(nx, ny)                            \
    REGISTER_OMP_CASE(nx, ny, 1);                        \
    REGISTER_OMP_CASE(nx, ny, 2);                        \
    REGISTER_OMP_CASE(nx, ny, 4);                        \
    REGISTER_OMP_CASE(nx, ny, 8);                        \
    REGISTER_OMP_CASE(nx, ny, 10);                       \
    REGISTER_OMP_CASE(nx, ny, 16);                       \
    REGISTER_OMP_CASE(nx, ny, 20)

REGISTER_GRID(512, 256);
REGISTER_GRID(1024, 512);
REGISTER_GRID(2048, 1024);
REGISTER_GRID(4096, 2048);

BENCHMARK_MAIN();