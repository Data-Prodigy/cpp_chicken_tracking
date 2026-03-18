#include "MaskMathGPU.cuh"
#include <math.h>
#include <device_launch_parameters.h>

__global__ void computeMasksKernelOpt(const float* __restrict__ d_coeffs,
    const float* __restrict__ d_proto,
    float* __restrict__ d_out_masks,
    int num_detections)
{
    int total_pixels = 25600;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int obj_idx = tid / total_pixels;
    int pixel_idx = tid % total_pixels;

    // Safety check
    if (obj_idx >= num_detections) return;

    // --- THE UPGRADE: SHARED MEMORY ---
    // Allocate 32 floats of ultra-fast cache per thread block
    __shared__ float shared_coeffs[32];

    // Note: Because 25600 is perfectly divisible by our block size (256), 
    // we are guaranteed that all 256 threads in this specific block are 
    // working on the EXACT SAME obj_idx.

    // Have the first 32 threads grab the 32 coefficients from slow Global VRAM
    if (threadIdx.x < 32) {
        shared_coeffs[threadIdx.x] = d_coeffs[obj_idx * 32 + threadIdx.x];
    }

    // Force all threads to wait here until the 32 floats are fully loaded into cache
    __syncthreads();

    // Now do the math!
    float val = 0.0f;
    for (int c = 0; c < 32; ++c) {
        // Read from ultra-fast shared memory instead of slow d_coeffs
        val += shared_coeffs[c] * d_proto[c * total_pixels + pixel_idx];
    }

    // Fast GPU Sigmoid Activation & Write out
    d_out_masks[obj_idx * total_pixels + pixel_idx] = 1.0f / (1.0f + expf(-val));
}

namespace MaskMathGPU {
    void computeMasks(float* d_coeffs, float* d_proto, float* d_out_masks, int num_detections, cudaStream_t stream) {
        if (num_detections == 0) return;

        int total_threads = num_detections * 25600;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        // Launch the optimized Kernel
        computeMasksKernelOpt << <blocks, threads_per_block, 0, stream >> > (d_coeffs, d_proto, d_out_masks, num_detections);
    }
}