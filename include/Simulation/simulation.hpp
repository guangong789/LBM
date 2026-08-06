#pragma once
#include <vector>
#include <numeric>
#include <GLFW/glfw3.h>
#include <Utils/utility.hpp>
#include <Visualization/texture.hpp>
#include <D2Q9/D2Q9_gpu.cuh>

struct ErrorMetrics {
    double l2;
    float max;
};

inline ErrorMetrics compareField(const std::vector<float>& cpu,
                                 const std::vector<float>& gpu) {
    assert(cpu.size() == gpu.size());

    float max_err = 0.f;
    double squared_error_sum = 0.0;

    for (size_t i = 0; i < cpu.size(); ++i) {
        const float err = std::abs(cpu[i] - gpu[i]);
        max_err = std::max(max_err, err);
        squared_error_sum += static_cast<double>(err) * err;
    }

    return {
        std::sqrt(squared_error_sum / static_cast<double>(cpu.size())),
        max_err
    };
}

inline void printValidationErrors(const std::vector<float>& rho_cpu,
                                  const std::vector<float>& ux_cpu,
                                  const std::vector<float>& uy_cpu,
                                  const std::vector<float>& rho_gpu,
                                  const std::vector<float>& ux_gpu,
                                  const std::vector<float>& uy_gpu) {
    const ErrorMetrics rho_error = compareField(rho_cpu, rho_gpu);
    const ErrorMetrics ux_error = compareField(ux_cpu, ux_gpu);
    const ErrorMetrics uy_error = compareField(uy_cpu, uy_gpu);

    std::cout << std::scientific
              << "rho: L2 = " << rho_error.l2 << ", Max = " << rho_error.max
              << " | ux: L2 = " << ux_error.l2 << ", Max = " << ux_error.max
              << " | uy: L2 = " << uy_error.l2 << ", Max = " << uy_error.max
              << std::endl;
}

template<typename FieldCPU, typename SolverCPU,typename FieldGPU, typename SolverGPU>
void runValidation() {
    const int nx = 2048, ny = 1024;
    const int totalSteps = 8000;

    FieldCPU cpu(nx, ny, Layout::SoA);
    FieldGPU gpu(nx, ny, Layout::SoA);

    setGPU();

    SolverCPU::initialize(cpu);
    SolverGPU::initialize(gpu);

    std::vector<float> rho_cpu, ux_cpu, uy_cpu;
    std::vector<float> rho_gpu, ux_gpu, uy_gpu;

    for (int step = 0; step < totalSteps; ++step) {
        SolverCPU::macro(cpu);
        SolverCPU::collide(cpu);
        SolverCPU::stream(cpu);
        SolverCPU::bounce_back(cpu);

        SolverGPU::macro(gpu);
        SolverGPU::collide(gpu);
        SolverGPU::stream(gpu);
        SolverGPU::bounce_back(gpu);

        if (step % 100 == 0) {
            // The simulation step updates f. Recompute macroscopic quantities so
            // the values being compared correspond to that updated state.
            SolverCPU::macro(cpu);
            SolverGPU::macro(gpu);

            fetchDensity(cpu, rho_cpu);
            fetchVelocity(cpu, ux_cpu, uy_cpu);
            fetchDensity(gpu, rho_gpu);
            fetchVelocity(gpu, ux_gpu, uy_gpu);

            std::cout << "[Step " << step << "] ";
            printValidationErrors(rho_cpu, ux_cpu, uy_cpu, rho_gpu, ux_gpu, uy_gpu);
        }
    }
    std::cout << "Validation finished." << std::endl;
}

void runSimulationGPU(GLFWwindow* window, LBMFieldGPU& field) {
    GLuint texID = initTexture(field.nx, field.ny);
    Shader rhoShader("docs/rhoShader/u.vs", "docs/rhoShader/u.fs");
    rhoShader.use();
    rhoShader.setInt("uTex", 0);
    Mesh quad(quadVertices);

    const int stepsPerFrame = 5;

    while(!glfwWindowShouldClose(window)) {
        for(int s=0; s<stepsPerFrame; ++s){
            D2Q9_gpu::macro(field);
            D2Q9_gpu::collide(field);
            D2Q9_gpu::stream(field);
            D2Q9_gpu::bounce_back(field);
        }

        updateTextureGPU(field, texID);

        glClear(GL_COLOR_BUFFER_BIT);
        rhoShader.use();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texID);
        quad.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    quad.cleanup();
    rhoShader.deleteProgram();
}
