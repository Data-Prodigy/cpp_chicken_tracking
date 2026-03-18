# PoultryHealth-AI: Real-Time TensorRT Segmentation & Tracking Pipeline

![Tracked Chickens](Chickens_gif_2.gif)

A low latency, production-grade computer vision pipeline designed to track, segment, and diagnose poultry health in real-time. 
Builtusing python and C++ with custom CUDA kernels, this system bypasses traditional CPU bottlenecks to achieve maximum hardware utilization on edge devices.

## System Architecture & Core Innovations
This project is built to solve the three primary bottlenecks of real-time video analytics: I/O Latency, PCIe Memory Bandwidth, and Temporal Neural Network Jitter.
Wrote a custom CUDA kernel (MaskMathGPU.cu) that computes the mask coefficients and Sigmoid activations directly in VRAM. (mask = sigmoid( Σ(coeff_i * proto_i) )).
By utilizing GPU __shared__ memory, the 32 mask coefficients are cached into L1, dropping global memory reads massively per frame and helping to speed up the system.

## Asynchronous I/O Threading
Since OpenCV's `cv::VideoCapture` is synchronous (i.e the camera stream waits for AI Inference to finish which may cause lags), I implemented a thread-safe class using `std::thread`, `std::mutex`, and `std::atomic` that pulls in camera frames via MSMF or FFMPEG endlessly in the background, ensuring the TensorRT engine always processes the absolute newest frame with zero blocking.

## TensorRT Int8 Calibration With GPU Preprocessing
To maximize inference speed, the YOLO11s model is quantized to INT8, basically making the trade-off between some accuracy for more speed. Int8Calibrator.h utilizes `cv::cuda::GpuMat` to perform letterboxing, padding, and normalization entirely on the GPU during the build process. The tracking system achieved a ~25 FPS.

## State Latching/Conviction Threshold
When the AI System receives livestream from my phone's camera (using DroidCam), it observes the chickens for a couple of consecutive frames and silently records their 'health state' based on fluctuating predictions from the yolo model, only when it crosses a certain confidence threshold (75%) does the monitoring system classify as unhealthy and locks that state for the chicken with that exact track ID. C++ ByteTrack Implementation gotten from [Junhui-Ng](https://github.com/junhui-ng/ByteTrack-CPP) 

## Project Structure
main.cpp: The core loop, Async Camera threading, ByteTrack integration, Conviction tracking, and HUD rendering.
YoloInference.cpp / .h: TensorRT engine execution, memory allocation, and CPU smart-cropping for masks.
MaskMathGPU.cu / .cuh: Custom CUDA C++ kernel for parallel mask multiplication and Sigmoid activation.
Int8Calibrator.h: Custom TensorRT Entropy Calibrator using OpenCV CUDA preprocessing.
send_alerts.py: Python SMTP client for external notification handling.

## Performance
Resolution: 1280x720 (Camera Input) -> 640x640 (Inference)
Tracking: ByteTrack (Kalman Filter + IoU Matching)
FPS: 25+ FPS on NVIDIA QUADRO T1000 GPU hardware with full mask rendering and tracking.

