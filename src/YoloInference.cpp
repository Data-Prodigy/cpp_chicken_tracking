#include "pch.h"
#include "YoloInference.h"
#include "MaskMathGPU.cuh"


YoloInference::YoloInference(const std::string& enginePath) {
    loadEngine(enginePath);
    context = engine->createExecutionContext();

    // Input
    inputSize = 1 * 3 * 640 * 640 * sizeof(float);

    // Output0: Detections (1 x 38 x 8400)
    output0Size = 1 * 38 * 8400 * sizeof(float);

    // Output1: Proto Masks (1 x 32 x 160 x 160)
    output1Size = 1 * 32 * 160 * 160 * sizeof(float);

    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], output0Size);
    cudaMalloc(&buffers[2], output1Size);

    cudaMalloc(&d_coeffs_buffer, 8400 * 32 * sizeof(float));
    cudaMalloc(&d_out_masks_buffer, 8400 * 25600 * sizeof(float));

    // Allocate CPU buffer exactly once
    cpu_output0.resize(1 * 38 * 8400);

    cudaStreamCreate(&stream);
}

void YoloInference::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(buffer.data(), size);
        std::cout << "Engine loaded successfully!\n";
    }
    else {
        std::cerr << "Error loading engine file!\n";
    }
}

std::vector<Detection> YoloInference::runInference(const cv::Mat& inputImage) {
    // 1. MANUAL PREPROCESSING (Letterbox Resize + Pad)
    float scale = std::min(640.0f / inputImage.cols, 640.0f / inputImage.rows);
    int nw = static_cast<int>(inputImage.cols * scale);
    int nh = static_cast<int>(inputImage.rows * scale);

    cv::Mat resized;
    cv::resize(inputImage, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);

    cv::Mat padded;
    int top = (640 - nh) / 2;
    int bottom = 640 - nh - top;
    int left = (640 - nw) / 2;
    int right = 640 - nw - left;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

    cudaMemcpyAsync(buffers[0], blob.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream);

    // 2. Inference
    context->setTensorAddress(engine->getIOTensorName(0), buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(1), buffers[1]);
    context->setTensorAddress(engine->getIOTensorName(2), buffers[2]);
    context->enqueueV3(stream);

    // 3. Postprocessing
    // --- THE MASSIVE FIX ---
    // We ONLY download output0 (the bounding boxes). 
    // We leave output1 (the proto masks) on the GPU for the CUDA Kernel.
    cudaMemcpyAsync(cpu_output0.data(), buffers[1], output0Size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Pass nullptr for output1 since we no longer use it on the CPU
    return postProcess(cpu_output0.data(), nullptr, inputImage.size());
}

std::vector<Detection> YoloInference::postProcess(float* output0, float* output1, cv::Size originalSize) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect_<float>> boxes640; // NEW: Store raw 640x640 boxes for the mask logic
    std::vector<std::vector<float>> maskCoefficients;

    const int num_classes = 2;
    const int num_masks = 32;
    const int rows = 8400;
    const int dimensions = 4 + num_classes + num_masks;

    // --- LETTERBOX MATH ---
    float scale = std::min(640.0f / originalSize.width, 640.0f / originalSize.height);
    float padX = (640.0f - originalSize.width * scale) / 2.0f;
    float padY = (640.0f - originalSize.height * scale) / 2.0f;

    cv::Mat output0Mat(dimensions, rows, CV_32F, output0);

    for (int i = 0; i < rows; ++i) {
        float maxScore = 0.0f;
        int classId = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = output0Mat.at<float>(4 + c, i);
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        if (maxScore > 0.1f) {
            float cx = output0Mat.at<float>(0, i);
            float cy = output0Mat.at<float>(1, i);
            float w = output0Mat.at<float>(2, i);
            float h = output0Mat.at<float>(3, i);

            // 1. Box in 640x640 space (Save this for the mask step)
            float x640 = cx - 0.5f * w;
            float y640 = cy - 0.5f * h;
            boxes640.push_back(cv::Rect_<float>(x640, y640, w, h));

            // 2. Box mapped back to the Original Video Frame (Subtract padding, divide by scale)
            int left = static_cast<int>((x640 - padX) / scale);
            int top = static_cast<int>((y640 - padY) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            boxes.push_back(cv::Rect(left, top, width, height));
            classIds.push_back(classId);
            confidences.push_back(maxScore);

            std::vector<float> coeffs;
            for (int m = 0; m < num_masks; ++m) {
                coeffs.push_back(output0Mat.at<float>(4 + num_classes + m, i));
            }
            maskCoefficients.push_back(coeffs);
        }
    }

 // 5. NMS (Non-Maximum Suppression)


// ... inside your postProcess function, after running NMS ...

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.1f, 0.65f, indices);

    int num_detections = indices.size();
    std::vector<Detection> results;

    if (num_detections > 0) {
        // 1. Gather surviving coefficients on the CPU
        std::vector<float> flat_coeffs(num_detections * 32);
        for (int i = 0; i < num_detections; ++i) {
            int original_idx = indices[i];
            memcpy(&flat_coeffs[i * 32], maskCoefficients[original_idx].data(), 32 * sizeof(float));
        }

        // 2. Upload coefficients to our PRE-ALLOCATED GPU buffer
        // Note: We only copy 'num_detections' worth of data, even though the buffer holds 8400.
        cudaMemcpyAsync(d_coeffs_buffer, flat_coeffs.data(), num_detections * 32 * sizeof(float), cudaMemcpyHostToDevice, stream);

        // 3. FIRE THE CUDA KERNEL!
        MaskMathGPU::computeMasks(d_coeffs_buffer, (float*)buffers[2], d_out_masks_buffer, num_detections, stream);

        // 4. Download the finished, sigmoid-activated masks back to CPU
        std::vector<float> flat_finished_masks(num_detections * 25600);
        cudaMemcpyAsync(flat_finished_masks.data(), d_out_masks_buffer, num_detections * 25600 * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // Wait for GPU to finish this frame's work
        cudaStreamSynchronize(stream);

        // Note: We REMOVED cudaFree here!

        // 5. Assign the masks to the tracking structs
        for (int i = 0; i < num_detections; ++i) {
            int original_idx = indices[i];
            Detection det;
            det.class_id = classIds[original_idx];
            det.confidence = confidences[original_idx];
            det.box = boxes[original_idx];

            cv::Mat mask160(160, 160, CV_32F, &flat_finished_masks[i * 25600]);

            cv::Rect_<float> b640 = boxes640[original_idx];
            int mx160 = std::max(0, (int)(b640.x / 4.0f) - 1);
            int my160 = std::max(0, (int)(b640.y / 4.0f) - 1);
            int mw160 = std::min(160 - mx160, (int)(b640.width / 4.0f) + 2);
            int mh160 = std::min(160 - my160, (int)(b640.height / 4.0f) + 2);

            cv::Rect crop160(mx160, my160, mw160, mh160);

            if (crop160.area() > 0) {
                cv::Mat tinyCrop = mask160(crop160);

                // STEP 1: threshold first
                cv::Mat binaryTiny;
                cv::threshold(tinyCrop, binaryTiny, 0.5, 1.0, cv::THRESH_BINARY);

                // STEP 2: resize AFTER threshold
                cv::Mat scaledCrop;
                cv::resize(binaryTiny, scaledCrop, cv::Size(mw160 * 4, mh160 * 4), 0, 0, cv::INTER_NEAREST);

                int offsetX = std::max(0, (int)b640.x - (mx160 * 4));
                int offsetY = std::max(0, (int)b640.y - (my160 * 4));
                int exactW = std::max(1, std::min(scaledCrop.cols - offsetX, (int)b640.width));
                int exactH = std::max(1, std::min(scaledCrop.rows - offsetY, (int)b640.height));

                cv::Rect exactCrop(offsetX, offsetY, exactW, exactH);
                det.boxMask = scaledCrop(exactCrop).clone();
            }
            results.push_back(det);
        }
    }
    return results;
}

YoloInference::~YoloInference() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);

    if (d_coeffs_buffer) cudaFree(d_coeffs_buffer);
    if (d_out_masks_buffer) cudaFree(d_out_masks_buffer);

    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
}