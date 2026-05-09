#include <Visualization/texture.hpp>
#include <cmath>

__global__ void buildTextureKernel(const float* ux, const float* uy, float* tex, int nx, int ny) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nx * ny) return;

    int x = id % nx;
    int y = id / nx;

    if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
        tex[id] = 0.0f;
        return;
    }

    int idx = y * nx + x;

    float dv_dx = (uy[y * nx + (x+1)] - uy[y * nx + (x-1)]) * 0.5f;
    float du_dy = (ux[(y+1)*nx + x] - ux[(y-1)*nx + x]) * 0.5f;

    float omega = dv_dx - du_dy;

    tex[id] = omega;
}