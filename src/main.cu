#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <Shader/shader.hpp>
#include <Mesh/mesh.hpp>
#include <Utils/global.hpp>
#include <Layout/lbmField_gpu.hpp>
#include <Layout/lbmField_cpu.hpp>
#include <Simulation/simulation.hpp>

#include <D2Q9/D2Q9_cpu.hpp>
#include <D2Q9/D2Q9_omp.hpp>
#include <D2Q9/D2Q9_gpu.cuh>

enum class Backend {
    SIMULATION,
    VALIDATE
};

constexpr Backend backend = Backend::SIMULATION;

int main() {
    GLContextRAII glctx(scrWidth, scrHeight, "Simulation");
    GLFWwindow* window = glctx.getWindow();

    if constexpr (backend == Backend::SIMULATION) {
        std::cout << "Running GPU Simulation" << std::endl;
        setGPU();
        const int nx = 2048;
        const int ny = 512;
        LBMFieldGPU field(nx, ny, Layout::SoA);
        D2Q9_gpu::initialize(field);
        runSimulationGPU(window, field);
    } else if constexpr (backend == Backend::VALIDATE) {
        std::cout << "Running validation (CPU Serial vs GPU)" << std::endl;
        runValidation<LBMFieldCPU, D2Q9_cpu, LBMFieldGPU, D2Q9_gpu>();
        return 0;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}