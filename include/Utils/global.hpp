#pragma once
#include <GLFW/glfw3.h>
#include <vector>

#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST_DEVICE
    #define HOST
    #define DEVICE
#endif

// window
inline GLFWwindow *window;
inline constexpr int scrWidth = 2048;
inline constexpr int scrHeight = 512;

// 全屏矩形顶点数据（x,y,u,v）
const std::vector<float> quadVertices{
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
     1.0f,  1.0f, 1.0f, 1.0f,

    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f,  1.0f, 1.0f, 1.0f,
    -1.0f,  1.0f, 0.0f, 1.0f
};