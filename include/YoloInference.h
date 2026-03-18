#pragma once
#include "pch.h" 

// Define a struct to hold our final result
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;           // Bounding Box
    cv::Mat boxMask;        // The pixel-perfect mask (instance segmentation)
};

class YoloInference {
public:
    // Constructor loads the engine
    YoloInference(const std::string& enginePath);
    ~YoloInference();

    // The main function: Image In -> Detections Out
    std::vector<Detection> runInference(const cv::Mat& inputImage);

private:
    // Helper to load engine file
    void loadEngine(const std::string& enginePath);

    // Helper to process raw output into Detections
    std::vector<Detection> postProcess(float* output0, float* output1, cv::Size originalSize);

    // TensorRT pointers
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    // GPU Memory Pointers
    void* buffers[3]; // 1 Input + 2 Outputs (Seg model has 2 outputs!)
    float* d_coeffs_buffer = nullptr;
    float* d_out_masks_buffer = nullptr;

    // Memory Sizes
    size_t inputSize;
    size_t output0Size; // Detections
    size_t output1Size; // Proto-masks

    // CUDA Stream for async execution
    cudaStream_t stream;

    // NEW: Persistent CPU buffer for the bounding boxes
    std::vector<float> cpu_output0;

    // Logger (Re-use your logger class or define a simple one)
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) std::cout << "[TRT] \n" << msg << "\n";
        }
    } logger;
}; 
