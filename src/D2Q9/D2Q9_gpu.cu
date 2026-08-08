#include <D2Q9/D2Q9_gpu.cuh>
#include <iostream>
#include <utility>
#include <vector>

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

__global__ void macro_collide_kernel(LBMFieldView v) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    const int cell = y * v.nx + x;
    const int cells = v.nx * v.ny;
    float distributions[9];

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int id = (v.layout == Layout::AoS)
            ? cell * 9 + k
            : k * cells + cell;
        distributions[k] = v.f[id];
    }

    if (v.is_cylinder[cell]) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int id = (v.layout == Layout::AoS)
                ? cell * 9 + k
                : k * cells + cell;
            v.f_next[id] = distributions[k];
        }
        return;
    }

    float rho = 0.f;
    float momentum_x = 0.f;
    float momentum_y = 0.f;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        rho += distributions[k];
        momentum_x += distributions[k] * cx[k];
        momentum_y += distributions[k] * cy[k];
    }

    if (rho < 1e-6f) rho = 1e-6f;

    const float ux = momentum_x / rho;
    const float uy = momentum_y / rho;
    const float u2 = ux * ux + uy * uy;

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int id = (v.layout == Layout::AoS)
            ? cell * 9 + k
            : k * cells + cell;
        const float cu = 3.f * (ux * cx[k] + uy * cy[k]);
        const float feq = w[k] * rho * (
            1.f + cu + 0.5f * cu * cu - 1.5f * u2
        );
        float updated = distributions[k]
            - (distributions[k] - feq) / D2Q9_gpu::tau;

        if (isnan(updated)) updated = feq;
        v.f_next[id] = updated;
    }
}

__global__ void macro_collide_stream_push_kernel(LBMFieldView v) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    const int cell = y * v.nx + x;
    const int cells = v.nx * v.ny;
    float distributions[9];

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const int id = (v.layout == Layout::AoS)
            ? cell * 9 + k
            : k * cells + cell;
        distributions[k] = v.f[id];
    }

    if (!v.is_cylinder[cell]) {
        float rho = 0.f;
        float momentum_x = 0.f;
        float momentum_y = 0.f;

        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            rho += distributions[k];
            momentum_x += distributions[k] * cx[k];
            momentum_y += distributions[k] * cy[k];
        }

        if (rho < 1e-6f) rho = 1e-6f;

        const float ux = momentum_x / rho;
        const float uy = momentum_y / rho;
        const float u2 = ux * ux + uy * uy;

        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            const float cu = 3.f * (ux * cx[k] + uy * cy[k]);
            const float feq = w[k] * rho * (
                1.f + cu + 0.5f * cu * cu - 1.5f * u2
            );
            float updated = distributions[k]
                - (distributions[k] - feq) / D2Q9_gpu::tau;

            if (isnan(updated)) updated = feq;
            distributions[k] = updated;
        }
    }

    // Push is the periodic inverse of the existing pull stream:
    // source (x, y, k) writes destination (x + cx[k], y + cy[k], k).
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        int destination_x = x + cx[k];
        int destination_y = y + cy[k];

        if (destination_x < 0) destination_x += v.nx;
        if (destination_x >= v.nx) destination_x -= v.nx;
        if (destination_y < 0) destination_y += v.ny;
        if (destination_y >= v.ny) destination_y -= v.ny;

        const int destination_cell =
            destination_y * v.nx + destination_x;
        const int destination_id = (v.layout == Layout::AoS)
            ? destination_cell * 9 + k
            : k * cells + destination_cell;
        v.f_next[destination_id] = distributions[k];
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

__global__ void stream_soa_v3_kernel(LBMFieldView v) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= v.nx || y >= v.ny) return;

    const int nx = v.nx;
    const int ny = v.ny;
    const int cell_count = nx * ny;

    const int xm = (x == 0) ? nx - 1 : x - 1;
    const int xp = (x == nx - 1) ? 0 : x + 1;
    const int ym = (y == 0) ? ny - 1 : y - 1;
    const int yp = (y == ny - 1) ? 0 : y + 1;

    const int row = y * nx;
    const int row_ym = ym * nx;
    const int row_yp = yp * nx;

    const int c = row + x;
    const int c_xm = row + xm;
    const int c_xp = row + xp;
    const int c_ym = row_ym + x;
    const int c_yp = row_yp + x;
    const int c_ym_xm = row_ym + xm;
    const int c_ym_xp = row_ym + xp;
    const int c_yp_xp = row_yp + xp;
    const int c_yp_xm = row_yp + xm;

    const float* __restrict__ src = v.f_next;
    float* __restrict__ dst = v.f;

    dst[c] = src[c];
    dst[cell_count + c] = src[cell_count + c_xm];
    dst[2 * cell_count + c] = src[2 * cell_count + c_ym];
    dst[3 * cell_count + c] = src[3 * cell_count + c_xp];
    dst[4 * cell_count + c] = src[4 * cell_count + c_yp];
    dst[5 * cell_count + c] = src[5 * cell_count + c_ym_xm];
    dst[6 * cell_count + c] = src[6 * cell_count + c_ym_xp];
    dst[7 * cell_count + c] = src[7 * cell_count + c_yp_xp];
    dst[8 * cell_count + c] = src[8 * cell_count + c_yp_xm];
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

__global__ void sparse_bounce_kernel(
    LBMFieldView v,
    const int* boundary_indices,
    int boundary_count
) {
    const int boundary_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (boundary_id >= boundary_count) return;

    const int c = boundary_indices[boundary_id];
    const int nx = v.nx;
    const int ny = v.ny;
    const int y = c / nx;

    auto idx = [&](int k) {
        return (v.layout == Layout::AoS) ? c * 9 + k : k * nx * ny + c;
    };

    // Preserve the original kernel's wall priority and direction swaps.
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

    // Every remaining entry satisfies the original is_cylinder predicate.
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

void D2Q9_gpu::macro_collide(LBMFieldGPU& field) {
    macro_collide(field, dim3(16,16));
}

void D2Q9_gpu::macro_collide(LBMFieldGPU& field, dim3 block) {
    const unsigned int threads_per_block = block.x * block.y * block.z;
    if (block.x == 0 || block.y == 0 || block.z != 1
        || threads_per_block > 1024) {
        std::cerr << "Skipping invalid Macro-Collide block configuration: "
                  << block.x << 'x' << block.y << 'x' << block.z << std::endl;
        return;
    }

    dim3 grid(
        (field.nx + block.x - 1) / block.x,
        (field.ny + block.y - 1) / block.y
    );
    macro_collide_kernel<<<grid,block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::macro_collide_stream_push(LBMFieldGPU& field) {
    macro_collide_stream_push(field, dim3(16, 16));
}

void D2Q9_gpu::macro_collide_stream_push(
    LBMFieldGPU& field,
    dim3 block
) {
    const unsigned int threads_per_block = block.x * block.y * block.z;
    if (block.x == 0 || block.y == 0 || block.z != 1
        || threads_per_block > 1024) {
        std::cerr << "Skipping invalid Macro-Collide Push-Stream block "
                  << "configuration: "
                  << block.x << 'x' << block.y << 'x' << block.z << std::endl;
        return;
    }

    dim3 grid(
        (field.nx + block.x - 1) / block.x,
        (field.ny + block.y - 1) / block.y
    );

    macro_collide_stream_push_kernel<<<grid, block>>>(field.view());
    CUDA_CHECK_KERNEL();

    // The launch captured the old pointer values. Swapping on the host makes
    // the streamed output the current state without copying device memory.
    std::swap(field.f, field.f_next);
}

void D2Q9_gpu::stream(LBMFieldGPU& field) {
    dim3 block(16,16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    stream_kernel<<<grid,block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::stream_v3(LBMFieldGPU& field) {
    if (field.layout != Layout::SoA) {
        stream(field);
        return;
    }

    dim3 block(32, 8);
    dim3 grid(
        (field.nx + block.x - 1) / block.x,
        (field.ny + block.y - 1) / block.y
    );
    stream_soa_v3_kernel<<<grid, block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::bounce_back(LBMFieldGPU& field) {
    dim3 block(16, 16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    bounce_kernel<<<grid, block>>>(field.view());
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::bounce_back_sparse(LBMFieldGPU& field) {
    if (field.boundary_count == 0) return;

    constexpr int threads_per_block = 256;
    const int blocks =
        (field.boundary_count + threads_per_block - 1)
        / threads_per_block;

    sparse_bounce_kernel<<<blocks, threads_per_block>>>(
        field.view(),
        field.boundary_indices,
        field.boundary_count
    );
    CUDA_CHECK_KERNEL();
}

void D2Q9_gpu::initialize(LBMFieldGPU& field) {
    dim3 block(16,16);
    dim3 grid((field.nx+15)/16, (field.ny+15)/16);
    initialize_kernel<<<grid,block>>>(field.view(), cylinder_r);
    CUDA_CHECK_KERNEL();

    const size_t cells = static_cast<size_t>(field.nx) * field.ny;
    std::vector<unsigned char> cylinder_mask(cells);
    CUDA_CHECK(cudaMemcpy(
        cylinder_mask.data(),
        field.is_cylinder,
        cells * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));

    std::vector<int> boundary_indices;
    boundary_indices.reserve(
        static_cast<size_t>(2) * field.nx
        + static_cast<size_t>(4 * cylinder_r * cylinder_r)
    );

    // The baseline kernel treats every set is_cylinder entry as bounce work;
    // this exact union preserves that numerical rule and removes wall overlap.
    for (int y = 0; y < field.ny; ++y) {
        for (int x = 0; x < field.nx; ++x) {
            const int cell = y * field.nx + x;
            const bool is_wall = y == 0 || y == field.ny - 1;
            if (is_wall || cylinder_mask[cell]) {
                boundary_indices.push_back(cell);
            }
        }
    }

    if (field.boundary_indices != nullptr) {
        CUDA_CHECK(cudaFree(field.boundary_indices));
        field.boundary_indices = nullptr;
    }

    field.boundary_count = static_cast<int>(boundary_indices.size());
    if (field.boundary_count > 0) {
        CUDA_CHECK(cudaMalloc(
            &field.boundary_indices,
            boundary_indices.size() * sizeof(int)
        ));
        CUDA_CHECK(cudaMemcpy(
            field.boundary_indices,
            boundary_indices.data(),
            boundary_indices.size() * sizeof(int),
            cudaMemcpyHostToDevice
        ));
    }
}
