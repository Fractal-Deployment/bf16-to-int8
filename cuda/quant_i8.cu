/* BF16/F32 → symmetric INT8 per block (B=64, zp=0, scale=amax/127).
 * Offline pin convert. Not train. train_ok=false. */
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    return __uint_as_float(u);
}

__global__ void k_quant_bf16(const uint16_t* in, int64_t n, int B, int8_t* q,
                             float* scale) {
    int blk = (int)blockIdx.x;
    int t = (int)threadIdx.x;
    int64_t start = (int64_t)blk * (int64_t)B;
    __shared__ float sh[64];
    float a = 0.f;
    float w = 0.f;
    int64_t i = start + (int64_t)t;
    int in_range = (t < B && i < n) ? 1 : 0;
    if (in_range) {
        w = bf16_to_f32(in[i]);
        a = fabsf(w);
    }
    sh[t] = a;
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    float am = sh[0];
    float s = (am > 0.f) ? (am / 127.f) : 1.f;
    if (t == 0) scale[blk] = s;
    __syncthreads();
    if (in_range) {
        int v = (int)rintf(w / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}

__global__ void k_quant_f32(const float* in, int64_t n, int B, int8_t* q,
                            float* scale) {
    int blk = (int)blockIdx.x;
    int t = (int)threadIdx.x;
    int64_t start = (int64_t)blk * (int64_t)B;
    __shared__ float sh[64];
    float a = 0.f;
    float w = 0.f;
    int64_t i = start + (int64_t)t;
    int in_range = (t < B && i < n) ? 1 : 0;
    if (in_range) {
        w = in[i];
        a = fabsf(w);
    }
    sh[t] = a;
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    float am = sh[0];
    float s = (am > 0.f) ? (am / 127.f) : 1.f;
    if (t == 0) scale[blk] = s;
    __syncthreads();
    if (in_range) {
        int v = (int)rintf(w / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}

static int launch(const void* h_in, size_t in_bytes, int64_t n, int B, int bf16,
                  int8_t* h_q, float* h_scale) {
    if (B != 64 || n < 1) return 2;
    int64_t nblk = (n + B - 1) / B;
    void* d_in = NULL;
    int8_t* d_q = NULL;
    float* d_s = NULL;
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) return 3;
    if (cudaMalloc(&d_q, (size_t)n) != cudaSuccess) {
        cudaFree(d_in);
        return 3;
    }
    if (cudaMalloc(&d_s, (size_t)nblk * sizeof(float)) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        return 3;
    }
    if (cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        cudaFree(d_s);
        return 4;
    }
    if (bf16)
        k_quant_bf16<<<(unsigned)nblk, 64>>>((const uint16_t*)d_in, n, B, d_q, d_s);
    else
        k_quant_f32<<<(unsigned)nblk, 64>>>((const float*)d_in, n, B, d_q, d_s);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_q);
        cudaFree(d_s);
        return 5;
    }
    int rc = 0;
    if (cudaMemcpy(h_q, d_q, (size_t)n, cudaMemcpyDeviceToHost) != cudaSuccess) rc = 4;
    if (cudaMemcpy(h_scale, d_s, (size_t)nblk * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        rc = 4;
    cudaFree(d_in);
    cudaFree(d_q);
    cudaFree(d_s);
    return rc;
}

extern "C" int quant_bf16_i8_block(const uint16_t* h_in, int64_t n, int B,
                                   int8_t* h_q, float* h_scale) {
    return launch(h_in, (size_t)n * 2u, n, B, 1, h_q, h_scale);
}

extern "C" int quant_f32_i8_block(const float* h_in, int64_t n, int B, int8_t* h_q,
                                  float* h_scale) {
    return launch(h_in, (size_t)n * 4u, n, B, 0, h_q, h_scale);
}
