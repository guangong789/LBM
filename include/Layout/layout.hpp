#pragma once
#include <cassert>
#include <Utils/global.hpp>

enum class Layout {
    AoS,   // cell-major
    SoA    // direction-major
};

/*
AoS : f[cell * Q + q]
SoA : f[q * nx * ny + cell]
*/

inline int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

HOST_DEVICE inline int index(int j, int i, int k, int nx, int ny, Layout layout) {
    if (layout == Layout::AoS) {
        return (i * nx + j) * 9 + k;
    } else {
        return k * (nx * ny) + (i * nx + j);
    }
}