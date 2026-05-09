#include <D2Q9/D2Q9_gpu.cuh>
#include <iostream>

__device__ __constant__ int cx[9] = {0,1,0,-1,0,1,-1,-1,1};
__device__ __constant__ int cy[9] = {0,0,1,0,-1,1,1,-1,-1};
__device__ __constant__ float w[9] = {
    4.f/9.f,
    1.f/9.f,1.f/9.f,1.f/9.f,1.f/9.f,
    1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f
};

// Kernel
__global__ void macro_kernel(LBMFieldView v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    int c = y * v.nx + x;
    float rho = 0.f, ux = 0.f, uy = 0.f;
    for (int k = 0; k < 9; ++k) {
        int id = (v.layout == Layout::AoS) ? c * 9 + k : k * v.nx * v.ny + c;
        float f = v.f[id];
        rho += f;
        ux += f * cx[k];
        uy += f * cy[k];
    }
    if (rho < 1e-6f) rho = 1e-6f;
    v.rho[c] = rho;
    v.ux[c] = ux / rho;
    v.uy[c] = uy / rho;
}

__global__ void collide_kernel(LBMFieldView v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    int c = y * v.nx + x;

    if (v.is_cylinder[c]) {
        for (int k = 0; k < 9; ++k) {
            int id = (v.layout == Layout::AoS) ? c * 9 + k : k * v.nx * v.ny + c;
            v.f_next[id] = v.f[id];
        }
        return;
    }

    float rho = v.rho[c];
    float ux = v.ux[c];
    float uy = v.uy[c];
    float u2 = ux * ux + uy * uy;

    for (int k = 0; k < 9; ++k) {
        int id = (v.layout == Layout::AoS) ? c * 9 + k : k * v.nx * v.ny + c;
        float cu = 3.f * (ux * cx[k] + uy * cy[k]);
        float feq = w[k] * rho * (1.f + cu + 0.5f * cu * cu - 1.5f * u2);
        float fval = v.f[id];
        v.f_next[id] = fval - (fval - feq) / D2Q9_gpu::tau;

        if (isnan(v.f_next[id])) v.f_next[id] = feq;
    }
}

__global__ void stream_kernel(LBMFieldView v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    int nx = v.nx;
    int ny = v.ny;
    int c  = y * nx + x;

    for (int k = 0; k < 9; ++k) {
        int xn = x - cx[k];
        int yn = y - cy[k];

        if (xn < 0) xn += nx;
        if (xn >= nx) xn -= nx;
        if (yn < 0) yn += ny;
        if (yn >= ny) yn -= ny;

        int src_c = yn * nx + xn;
        int src = (v.layout == Layout::AoS) ? src_c * 9 + k : k * nx * ny + src_c;
        int dst = (v.layout == Layout::AoS) ? c * 9 + k : k * nx * ny + c;
        v.f[dst] = v.f_next[src];
    }
}

__global__ void bounce_kernel(LBMFieldView v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    int nx = v.nx, ny = v.ny;
    int c  = y * nx + x;

    auto idx = [&](int k) {
        return (v.layout == Layout::AoS) ? c*9 + k : k*nx*ny + c;
    };

    // top/bottom
    if (y == 0 || y == ny - 1) {

        float f2 = v.f[idx(2)];
        float f4 = v.f[idx(4)];
        float f5 = v.f[idx(5)];
        float f7 = v.f[idx(7)];
        float f6 = v.f[idx(6)];
        float f8 = v.f[idx(8)];

        v.f[idx(2)] = f4;
        v.f[idx(4)] = f2;

        v.f[idx(5)] = f7;
        v.f[idx(7)] = f5;

        v.f[idx(6)] = f8;
        v.f[idx(8)] = f6;

        return;
    }

    // ylinder
    if (v.is_cylinder[c]) {
        float f1 = v.f[idx(1)];
        float f3 = v.f[idx(3)];
        float f2 = v.f[idx(2)];
        float f4 = v.f[idx(4)];
        float f5 = v.f[idx(5)];
        float f7 = v.f[idx(7)];
        float f6 = v.f[idx(6)];
        float f8 = v.f[idx(8)];

        v.f[idx(1)] = f3;
        v.f[idx(3)] = f1;

        v.f[idx(2)] = f4;
        v.f[idx(4)] = f2;

        v.f[idx(5)] = f7;
        v.f[idx(7)] = f5;

        v.f[idx(6)] = f8;
        v.f[idx(8)] = f6;
    }
}

__global__ void initialize_kernel(LBMFieldView view, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= view.nx || y >= view.ny) return;

    int cell = y * view.nx + x;
    int cx0 = view.nx / 4;
    int cy0 = view.ny / 2;
    int dx = x - cx0;
    int dy = y - cy0;
    view.is_cylinder[cell] = (dx*dx + dy*dy <= r*r);

    float ux = D2Q9_gpu::ux0;
    float uy = 0.f;
    float rho = D2Q9_gpu::rho0;
    float u2 = ux*ux + uy*uy;

    for (int k = 0; k < 9; ++k) {
        float cu = 3.f * (ux * cx[k] + uy * cy[k]);
        float feq = w[k] * rho * (1.f + cu + 0.5f*cu*cu - 1.5f*u2);
        int id = (view.layout == Layout::AoS) ? cell*9 + k : k*view.nx*view.ny + cell;
        view.f[id] = feq;
    }

    view.rho[cell] = rho;
    view.ux[cell]  = ux;
    view.uy[cell]  = uy;
}

// Host
void D2Q9_gpu::macro(LBMFieldGPU& field) {
    dim3 block(16,16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    macro_kernel<<<grid,block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::collide(LBMFieldGPU& field) {
    dim3 block(16,16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    collide_kernel<<<grid,block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::stream(LBMFieldGPU& field) {
    dim3 block(16,16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    stream_kernel<<<grid,block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::bounce_back(LBMFieldGPU& field) {
    dim3 block(16, 16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    bounce_kernel<<<grid, block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::initialize(LBMFieldGPU& field) {
    dim3 block(16,16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    initialize_kernel<<<grid,block>>>(field.view(), cylinder_r);
    CUDA_CHECK_KERNEL();
}