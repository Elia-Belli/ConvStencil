#include "2d_utils.h"

#ifdef __INTELLISENSE__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <mma.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include "precision.h"
// #include "../utils.h"

#define DEBUG

using namespace nvcuda;

#define BLOCK_SIZE_ROW 32
#define BLOCK_SIZE_COL 128  // 64
#define HALO 3
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)    // 128 + 6 = 134
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + HALO * 2)    // 32 + 6  = 38
#define PAD 2
#define SM_SIZE_COL (7 * D_BLOCK_SIZE_ROW + PAD)    // 7 * 38 + 2  = 266
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 8)          // 134 / 8     = 16
#define UNIT_LENGTH 7
#define TENSOR_CORE_M 16 // 8
#define TENSOR_CORE_N 16 // 8
#define TENSOR_CORE_K 8 // 4
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define WARP_PER_BLOCK 8
// #define ACCS_PER_WARP (BLOCK_SIZE_COL * BLOCK_SIZE_ROW / 64 / WARP_PER_BLOCK)
#define MMA_NUM 7 // 13
#define ceild(n,d)	(((n)-1)/(d) + 1)

__constant__ real_t param_matrix_d[2 * MMA_NUM * TENSOR_CORE_M * TENSOR_CORE_K];


__global__ void kernel2d_fp32 (const float * __restrict__ in, float * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    
    __shared__ __align__(32) float sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
    int warp_id = threadIdx.x / 32;

    // Load data into shared memory using lookup tables
    /*
        Data is loaded from global memory, in which resides the original input array.
        When loading into shared memory, we use lookup tables to apply the s2r layout.
        Data in shared memory has the stencil2row layout.
    */
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
    }
    __syncthreads();


    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * TENSOR_CORE_M * TENSOR_CORE_K, TENSOR_CORE_M);
        wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + (MMA_NUM + i) * TENSOR_CORE_M * TENSOR_CORE_K, TENSOR_CORE_M);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> in_frag;
    for (int col = warp_id * 28; col < warp_id * 28 + 28; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + (compute_idx * TENSOR_CORE_K + col), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + (compute_idx * TENSOR_CORE_K + col), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}


/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer (length 49)
 * 
*/
void gpu_box_2d1r(const real_t * __restrict__ in, real_t * __restrict__ out, const real_t * __restrict__ params, const int times, const int input_m, const int input_n) {
    real_t param_matrix_h[2][MMA_NUM * TENSOR_CORE_M * TENSOR_CORE_K] = {0.0};

    // Build Weight Matrix A
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    int idx = (i * UNIT_LENGTH + j) * TENSOR_CORE_M + col;
                    param_matrix_h[0][idx] = params[i * UNIT_LENGTH + j - col];
                    param_matrix_h[0][idx+8] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    // Build Weight Matrix B
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col - 8) {
                    int idx = (i * UNIT_LENGTH + j) * TENSOR_CORE_M + col;
                    param_matrix_h[1][idx] = params[i * UNIT_LENGTH + j - col + 15];
                    param_matrix_h[1][idx-8] = params[i * UNIT_LENGTH + j - col + 15];
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, sizeof(param_matrix_h)));

    #ifdef DEBUG

    std::cout << "[Stencil Kernel]" << std::endl;
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 7; j++){
            std::cout << params[i * 7 + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n[Weight Matrix A]" << std::endl;
    for (int i = 0; i < MMA_NUM; i++) {
        int mma_offset = i* TENSOR_CORE_M * TENSOR_CORE_K;
        for(int j=0; j < TENSOR_CORE_K; j++){
            for(int k=0; k < TENSOR_CORE_M; k++){
                std::cout << param_matrix_h[0][mma_offset + j * TENSOR_CORE_M + k] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\n[Weight Matrix B]" << std::endl;
    for (int i = 0; i < MMA_NUM; i++) {
        int mma_offset = i * TENSOR_CORE_M * TENSOR_CORE_K;
        for(int j = 0; j < TENSOR_CORE_K; j++){
            for(int k = 0; k < TENSOR_CORE_M; k++){
                std::cout << param_matrix_h[1][mma_offset + j * TENSOR_CORE_M + k] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    
    #endif

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO + 2;
    const size_t array_size = rows * cols * sizeof(real_t);
    real_t *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // Lookup tables (with linearized indices)
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            // Stencil2row Matrix A
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            // Stencil2row Matrix B
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }

    #ifdef DEBUG

    std::cout << "\nSharedmem[0]" << std::endl;
    float debug_sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL] = {0.0};

    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++)
    {
        for(int j = 0; j < D_BLOCK_SIZE_COL; j++) 
        {
            debug_sharedmem[0][lookup_table1_h[i][j]] = in[IDX(i, j, cols)];
            debug_sharedmem[1][lookup_table2_h[i][j]] = in[IDX(i, j, cols)];
        }
    }

    for (int i = 0; i < SM_SIZE_ROW; i++)
    {
        for(int j = 0; j < SM_SIZE_COL; j++) 
        {
            std::cout << debug_sharedmem[0][i * SM_SIZE_COL + j] << " ";
            //sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++)
    {
        for(int j = 0; j < D_BLOCK_SIZE_COL; j++) 
        {
            std::cout << in[IDX(i, j, cols)] << " ";
            //sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
        }
        std::cout << std::endl;
    }

    #endif

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        #ifdef USE_DOUBLE_PRECISION
            CUDAKERNELCHECK((kernel2d_fp64<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
        #endif
        #ifdef USE_FLOAT_PRECISION
            CUDAKERNELCHECK((kernel2d_fp32<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
        #endif
        }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "ConvStencil(2D): " << std::endl;
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    printf("GStencil/s = %f\n", ((double)input_m * input_n * times * 3) / secs / 1e9);
    
    std::ofstream csv("logs/logs.csv", std::ios::app);
    csv << "ConvStencil(2D),star_2d1r," << input_m << "," << times << "," << precision_name(out[0]) << ","
        << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "," 
        << ((double)input_m * input_n * times * 3) / secs / 1e9 << std::endl;


    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}


/*
__global__ void kernel2d_fp64 (const double * __restrict__ in, double * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
    }
    __syncthreads();


    int warp_id = threadIdx.x / 32;

    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 52 * 8 + i * 32, 8);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    for (int col = warp_id * 28; col < warp_id * 28 + 28; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}
*/