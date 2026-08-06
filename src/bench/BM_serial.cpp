#include <benchmark/benchmark.h>
#include <Layout/lbmField_cpu.hpp>
#include <Layout/layout.hpp>
#include <D2Q9/D2Q9_cpu.hpp>

template <typename Solver>
static void RunSerialBenchmark(
    benchmark::State& state,
    Layout layout
) {
    const int nx = static_cast<int>(state.range(0));
    const int ny = static_cast<int>(state.range(1));

    LBMFieldCPU field(nx, ny, layout);
    Solver::initialize(field);

    // Warm-up，不计入正式测试时间
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
}

static void BM_Serial_AoS(benchmark::State& state) {
    RunSerialBenchmark<D2Q9_cpu>(state, Layout::AoS);
}

static void BM_Serial_SoA(benchmark::State& state) {
    RunSerialBenchmark<D2Q9_cpu>(state, Layout::SoA);
}

#define REGISTER_SERIAL_CASE(benchmark_name, nx, ny) \
    BENCHMARK(benchmark_name)                        \
        ->Args({nx, ny})                             \
        ->UseRealTime()                              \
        ->Unit(benchmark::kMillisecond)

BENCHMARK(BM_Serial_AoS)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Serial_SoA)
    ->Args({512, 256})
    ->Args({1024, 512})
    ->Args({2048, 1024})
    ->Args({4096, 2048})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();