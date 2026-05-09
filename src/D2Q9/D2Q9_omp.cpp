#include <D2Q9/D2Q9_omp.hpp>
#include <Layout/layout.hpp>
#include <cmath>
#include <algorithm>

void D2Q9_cpu_omp::macro(LBMFieldCPU& field) {
    int nx = field.nx; 
    int ny = field.ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < ny; ++i){
        for(int j = 0; j < nx; ++j){
            float rho_ij = 0.f, ux_ij = 0.f, uy_ij = 0.f;
            for(int k = 0; k < Q; ++k){
                int id = index(j, i, k, nx, ny, field.layout);
                float fv = field.f[id];
                rho_ij += fv;
                ux_ij += fv*cx[k];
                uy_ij += fv*cy[k];
            }
            int flat = i*nx + j;
            if(rho_ij < 1e-6f) rho_ij = 1e-6f;
            field.rho[flat] = rho_ij;
            field.ux[flat] = ux_ij / rho_ij;
            field.uy[flat] = uy_ij / rho_ij;
        }
    }
}

void D2Q9_cpu_omp::collide(LBMFieldCPU& field) {
    int nx = field.nx;
    int ny = field.ny;

    #pragma omp parallel for collapse(2) schedule(static, 32)
    for(int i = 0; i < ny; ++i) {
        for(int j = 0; j < nx; ++j) {
            int flat = i * nx + j;

            if(field.is_cylinder[flat]){
                for(int k = 0; k < Q; ++k) {
                    field.f_next[index(j, i, k, nx, ny, field.layout)] = 
                        field.f[index(j, i, k, nx, ny, field.layout)];
                }
                continue;
            }

            float rho_ij = field.rho[flat];
            float ux_ij  = field.ux[flat];
            float uy_ij  = field.uy[flat];

            float u2 = ux_ij*ux_ij + uy_ij*uy_ij;

            for(int k = 0; k < Q; ++k) {
                int id = index(j, i, k, nx, ny, field.layout);
                float fval = field.f[id];

                float eu = ux_ij * cx[k] + uy_ij * cy[k];
                float cu = 3.f * eu;

                float feq = w[k] * rho_ij * (1 + cu + 0.5f*cu*cu - 1.5f*u2);

                field.f_next[id] = fval - (fval - feq) / tau;
                if(std::isnan(field.f_next[id])) field.f_next[id] = feq;
            }
        }
    }
}

void D2Q9_cpu_omp::stream(LBMFieldCPU& field) {
    int nx = field.nx;
    int ny = field.ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for(int y = 0; y < ny; ++y) {
        for(int x = 0; x < nx; ++x) {
            for(int k = 0; k < Q; ++k) {
                int xn = x - cx[k];
                int yn = y - cy[k];

                if (xn < 0)   xn += nx;
                if (xn >= nx) xn -= nx;
                if (yn < 0)   yn += ny;
                if (yn >= ny) yn -= ny;

                int src = index(xn, yn, k, nx, ny, field.layout);
                int dst = index(x,  y,  k, nx, ny, field.layout);
                field.f[dst] = field.f_next[src];
            }
        }
    }
}

void D2Q9_cpu_omp::bounce_back(LBMFieldCPU& field) {
    int nx = field.nx;
    int ny = field.ny;

    #pragma omp parallel for schedule(static)
    for(int x = 0; x < nx; ++x) {
        int yb = 0;
        int yt = ny - 1;

        auto swap_dirs = [&](int y, int a, int b){ 
            std::swap(field.f[index(x, y, a, nx, ny, field.layout)], 
                     field.f[index(x, y, b, nx, ny, field.layout)]);
        };

        swap_dirs(yb, 2, 4); swap_dirs(yb, 5, 7); swap_dirs(yb, 6, 8);
        swap_dirs(yt, 2, 4); swap_dirs(yt, 5, 7); swap_dirs(yt, 6, 8);
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for(int y = 0; y < ny; ++y) {
        for(int x = 0; x < nx; ++x) {
            int flat = y*nx + x;
            if(!field.is_cylinder[flat]) continue;

            auto swap_dirs = [&](int a, int b){
                std::swap(field.f[index(x, y, a, nx, ny, field.layout)], 
                         field.f[index(x, y, b, nx, ny, field.layout)]);
            };

            swap_dirs(1, 3);
            swap_dirs(2, 4);
            swap_dirs(5, 7);
            swap_dirs(6, 8);
        }
    }
}

void D2Q9_cpu_omp::initialize(LBMFieldCPU& field) {
    int nx = field.nx;
    int ny = field.ny;
    int cx0 = field.nx / 8;
    int cy0 = field.ny / 2;

    #pragma omp parallel for collapse(2)
    for(int y = 0; y < ny; ++y) {
        for(int x = 0; x < nx; ++x) {
            int flat = y * nx + x;

            int dx = x - cx0;
            int dy = y - cy0;
            bool is_cyl = (dx*dx + dy*dy <= cylinder_r * cylinder_r);

            field.is_cylinder[flat] = is_cyl;

            float ux = D2Q9_cpu_omp::ux0;
            float uy = 0.f;
            float rho = D2Q9_cpu_omp::rho0;

            float u2 = ux*ux + uy*uy;

            for(int k = 0; k < Q; ++k) {
                float cu = 3.f * (ux * cx[k] + uy * cy[k]);
                float feq = w[k] * rho * (1.f + cu + 0.5f * cu * cu - 1.5f * u2);
                int id = index(x, y, k, nx, ny, field.layout);
                field.f[id] = feq;
            }

            field.rho[flat] = rho;
            field.ux[flat]  = ux;
            field.uy[flat]  = uy;
        }
    }
}