#pragma once
#include <Layout/lbmField_gpu.hpp>

struct D2Q9_gpu {
    // lattice
    static constexpr int Q = 9;

    // initial condition
    static constexpr float rho0 = 1.f;
    static constexpr float ux0  = 0.15f;
    static constexpr float uy0  = 0.f;

    // relaxation
    static constexpr float tau = 0.6f;

    // cylinder parameters
    static constexpr int cylinder_r  = 20;

    // LBM steps
    static void macro(LBMFieldGPU& field);
    static void collide(LBMFieldGPU& field);
    static void macro_collide(LBMFieldGPU& field);
    static void macro_collide(LBMFieldGPU& field, dim3 block);
    static void macro_collide_stream_push(LBMFieldGPU& field);
    static void macro_collide_stream_push(LBMFieldGPU& field, dim3 block);
    static void stream(LBMFieldGPU& field);
    static void stream_v3(LBMFieldGPU& field);
    static void bounce_back(LBMFieldGPU& field);
    static void bounce_back_sparse(LBMFieldGPU& field);

    // initialization
    static void initialize(LBMFieldGPU& field);
};
