#pragma once
#include <vector>
#include <cassert>
#include <Layout/lbmFieldView.cuh>
#include <Layout/layout.hpp>

struct LBMFieldCPU {
    std::vector<float> f;
    std::vector<float> f_next;
    std::vector<float> rho;
    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<unsigned char> is_cylinder;

    int nx = 0;
    int ny = 0;
    Layout layout;

    LBMFieldCPU(int nx_, int ny_, Layout layout_) : nx(nx_), ny(ny_), layout(layout_) {
        int cells = nx * ny;
        size_t total_f = static_cast<size_t>(cells) * 9;

        f.resize(total_f);
        f_next.resize(total_f);

        if (reinterpret_cast<uintptr_t>(f.data()) % 64 != 0) {
            std::vector<float> aligned_f;
            aligned_f.reserve(total_f + 16);
        }

        rho.resize(cells);
        ux.resize(cells);
        uy.resize(cells);
        is_cylinder.resize(cells);
    }

    LBMFieldView view() {
        return LBMFieldView(f.data(), f_next.data(), rho.data(), ux.data(), uy.data(), is_cylinder.data(), nx, ny, layout);
    }
};