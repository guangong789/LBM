#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include <D2Q9/D2Q9_gpu.cuh>
#include <Layout/lbmField_gpu.hpp>
#include <Layout/layout.hpp>
#include <Utils/Helper/cudaTools.cuh>

namespace {

struct ProfileConfig {
    int nx = 4096;
    int ny = 2048;
    int warmup_steps = 20;
    int profile_steps = 10;
    Layout layout = Layout::SoA;
};

void printUsage(const char* program) {
    std::cout << "Usage: " << program
              << " [--nx N] [--ny N] [--warmup N] [--steps N] [--layout aos|soa]\n";
}

ProfileConfig parseArgs(int argc, char** argv) {
    ProfileConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << '\n';
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }

        const std::string value = argv[++i];
        if (arg == "--nx") config.nx = std::stoi(value);
        else if (arg == "--ny") config.ny = std::stoi(value);
        else if (arg == "--warmup") config.warmup_steps = std::stoi(value);
        else if (arg == "--steps") config.profile_steps = std::stoi(value);
        else if (arg == "--layout") {
            if (value == "aos") config.layout = Layout::AoS;
            else if (value == "soa") config.layout = Layout::SoA;
            else {
                std::cerr << "Unsupported layout: " << value << '\n';
                std::exit(EXIT_FAILURE);
            }
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (config.nx <= 0 || config.ny <= 0 || config.warmup_steps < 0 || config.profile_steps <= 0) {
        std::cerr << "Grid dimensions and --steps must be positive; --warmup cannot be negative.\n";
        std::exit(EXIT_FAILURE);
    }
    return config;
}

void advance(LBMFieldGPU& field) {
    D2Q9_gpu::macro(field);
    D2Q9_gpu::collide(field);
    D2Q9_gpu::stream(field);
    D2Q9_gpu::bounce_back(field);
}

} // namespace

int main(int argc, char** argv) {
    const ProfileConfig config = parseArgs(argc, argv);
    setGPU();

    LBMFieldGPU field(config.nx, config.ny, config.layout);
    D2Q9_gpu::initialize(field);

    for (int step = 0; step < config.warmup_steps; ++step) {
        advance(field);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Profiling " << config.profile_steps << " step(s), grid "
              << config.nx << 'x' << config.ny << ", layout "
              << (config.layout == Layout::SoA ? "SoA" : "AoS") << '\n';

    CUDA_CHECK(cudaProfilerStart());
    for (int step = 0; step < config.profile_steps; ++step) {
        advance(field);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());

    return EXIT_SUCCESS;
}
