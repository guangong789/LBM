#pragma once
#include <array>
#include <vector>
#include <Layout/lbmField_cpu.hpp>

struct D2Q9_cpu {
    static constexpr int Q = 9;

    static constexpr float rho0 = 1.f;
    static constexpr float ux0  = 0.15f;
    static constexpr float uy0  = 0.f;

    // discrete velocities
    static constexpr std::array<int, Q> cx = {0, 1, 0, -1, 0, 1, -1, -1,  1};
    static constexpr std::array<int, Q> cy = {0, 0, 1,  0,-1, 1,  1, -1, -1};

    // weights
    static constexpr std::array<float, Q> w = {
        4.f/9.f,
        1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
        1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f
    };

    // relaxation time
    static constexpr float tau = 0.6f;

    // cylinder
    static constexpr int cylinder_r  = 20;

    // steps
    static void macro(LBMFieldCPU& field);
    static void collide(LBMFieldCPU& field);
    static void stream(LBMFieldCPU& field);
    static void bounce_back(LBMFieldCPU& field);

    // initialization
    static void initialize(LBMFieldCPU& field);
};