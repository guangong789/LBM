#include <benchmark/benchmark.h>
#include <Layout/lbmField_cpu.hpp>
#include <Layout/layout.hpp>
#include <D2Q9/D2Q9_cpu.hpp>

template<typename Solver>
static void RunSerialBenchmark(benchmark::State& state, Layout layout) {
    int nx = state.range(0);
    int ny = state.range(1);

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
}

static void BM_Serial_AoS(benchmark::State& state) {
    RunSerialBenchmark<D2Q9_cpu>(state, Layout::AoS);
}

static void BM_Serial_SoA(benchmark::State& state) {
    RunSerialBenchmark<D2Q9_cpu>(state, Layout::SoA);
}

BENCHMARK(BM_Serial_AoS)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Serial_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();