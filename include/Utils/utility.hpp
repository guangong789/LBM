#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <Layout/lbmField_gpu.hpp>
#include <type_traits>

template<typename T>
struct is_gpu_field : std::false_type {};

template<>
struct is_gpu_field<LBMFieldGPU> : std::true_type {};

class GLContextRAII {
public:
    GLContextRAII(int width, int height, const char* title);
    ~GLContextRAII();

    GLFWwindow* getWindow() const { return window; }
    void processInput();

private:
    GLFWwindow* window = nullptr;

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
};