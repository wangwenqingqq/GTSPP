#ifndef MLP_CONSTANT_CUH
#define MLP_CONSTANT_CUH

#include <cuda_runtime.h>

// =========================================================

// =========================================================
extern __device__ __constant__ float input_scale[3];
extern __device__ __constant__ float c_W1[3 * 8];
extern __device__ __constant__ float c_b1[8];
extern __device__ __constant__ float c_W2[8];
extern __device__ __constant__ float c_b2; 


void upload_mlp_constants(float* h_scale, float* h_W1, float* h_b1, float* h_W2, float* h_b2);

#endif // MLP_CONSTANT_CUH