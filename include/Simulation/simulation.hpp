#pragma once
#include <vector>
#include <numeric>
#include <GLFW/glfw3.h>
#include <Utils/utility.hpp>
#include <Visualization/texture.hpp>
#include <D2Q9/D2Q9_gpu.cuh>

inline void compareVelocity(const std::vector<float>& ux_cpu, const std::vector<float>& uy_cpu, const std::vector<float>& ux_gpu, const std::vector<float>& uy_gpu) {
    int n = ux_cpu.size();

    float max_err = 0.f;
    double l2 = 0.0;

    for (int i = 0; i < n; ++i) {
        float du = ux_cpu[i] - ux_gpu[i];
        float dv = uy_cpu[i] - uy_gpu[i];

        float err = std::sqrt(du*du + dv*dv);

        max_err = std::max(max_err, err);
        l2 += err * err;
    }

    l2 = std::sqrt(l2 / n);

    std::cout << std::scientific << "L2 = " << l2 << ", Max = " << max_err << std::endl;
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

    std::vector<float> ux_cpu, uy_cpu;
    std::vector<float> ux_gpu, uy_gpu;

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

            fetchVelocity(cpu, ux_cpu, uy_cpu);
            fetchVelocity(gpu, ux_gpu, uy_gpu);

            std::cout << "[Step " << step << "] ";
            compareVelocity(ux_cpu, uy_cpu, ux_gpu, uy_gpu);
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