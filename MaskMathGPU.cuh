#pragma once
#include <cuda_runtime.h>

namespace MaskMathGPU {
    // Launches the GPU thread grid to compute masks for all chickens in parallel
    void computeMasks(float* d_coeffs, float* d_proto, float* d_out_masks, int num_detections, cudaStream_t stream);
}