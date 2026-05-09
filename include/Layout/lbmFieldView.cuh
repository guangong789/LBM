#pragma once
#include <Layout/layout.hpp>

struct LBMFieldView {
    float* f;
    float* f_next;
    float* rho;
    float* ux;
    float* uy;
    unsigned char* is_cylinder;

    int nx;
    int ny;

    Layout layout;

    LBMFieldView() = default;

    HOST_DEVICE
    LBMFieldView(
        float* f_,
        float* f_next_,
        float* rho_,
        float* ux_,
        float* uy_,
        unsigned char*  is_cylinder_,
        int nx_,
        int ny_,
        Layout layout_) :
            f(f_),
            f_next(f_next_),
            rho(rho_),
            ux(ux_),
            uy(uy_),
            is_cylinder(is_cylinder_),
            nx(nx_),
            ny(ny_),
            layout(layout_) {}
};