#pragma once
#include "pch.h"


class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(int batchSize, int inputW, int inputH, const std::string& imgDir, const std::string& calibTableName)
        : mBatchSize(batchSize), mInputW(inputW), mInputH(inputH), mImgDir(imgDir), mCalibTableName(calibTableName) {

        // 1. Load file paths
        for (const auto& entry : std::filesystem::directory_iterator(mImgDir)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg") {
                mFilePaths.push_back(entry.path().string());
            }
        }

        // Shuffle to get a good random sample of data
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(mFilePaths.begin(), mFilePaths.end(), g);

        // 2. Allocate GPU Memory for the ENTIRE BATCH (Linear memory)
        // Size = Batch * 3 Channels * H * W * sizeof(float)
        mInputCount = batchSize * 3 * inputW * inputH;
        cudaError_t err = cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA Malloc Failed: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    }

    virtual ~Int8Calibrator() {
        cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mCursor >= mFilePaths.size()) return false;

        std::cout << "[Calibrator] Processing batch starting at image " << mCursor << "..." << std::endl;

        // Process each image in the batch
        for (int i = 0; i < mBatchSize; i++) {
            if (mCursor >= mFilePaths.size()) break; // Handle partial last batch if needed

            // 1. Load Image to CPU (OpenCV doesn't support loading directly to GPU well)
            cv::Mat cpuImg = cv::imread(mFilePaths[mCursor++]);
            if (cpuImg.empty()) {
                std::cerr << "Skipping corrupt image." << std::endl;
                continue;
            }

            // 2. Upload to GPU
            cv::cuda::GpuMat gpuImg;
            gpuImg.upload(cpuImg);

            // 3. LETTERBOX RESIZE (Aspect Ratio Preserved)
            float scale = std::min((float)mInputW / gpuImg.cols, (float)mInputH / gpuImg.rows);
            int nw = (int)(gpuImg.cols * scale);
            int nh = (int)(gpuImg.rows * scale);

            cv::cuda::GpuMat resized;
            cv::cuda::resize(gpuImg, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

            // 4. Pad with Grey (114) to fit 640x640
            cv::cuda::GpuMat padded;
            int top = (mInputH - nh) / 2;
            int bottom = mInputH - nh - top;
            int left = (mInputW - nw) / 2;
            int right = mInputW - nw - left;

            // value 114 is standard YOLO padding color
            cv::cuda::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

            // 5. Convert BGR -> RGB & Normalize 0-1 (Float)
            cv::cuda::GpuMat floatImg;
            padded.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f); // 0-255 -> 0.0-1.0
            cv::cuda::cvtColor(floatImg, floatImg, cv::COLOR_BGR2RGB);

            // 6. Split Channels (HWC -> CHW)
            std::vector<cv::cuda::GpuMat> channels(3);
            cv::cuda::split(floatImg, channels);

            // 7. Copy directly into the big Batch Buffer at the correct offset
            // Memory Layout: [Batch0_R, Batch0_G, Batch0_B, Batch1_R, ...]
            // Offset for this image = i * (3 * H * W)
            size_t planeSize = mInputH * mInputW;
            size_t imgOffset = i * 3 * planeSize;

            for (int c = 0; c < 3; c++) {
                // Determine where this channel goes in the linear buffer
                float* targetPtr = (float*)mDeviceInput + imgOffset + (c * planeSize);

                // Copy GpuMat data to the linear buffer
                // GpuMat might be padded, so we copy row by row or use cudaMemcpy2D if needed.
                // However, since we allocated mDeviceInput linearly, we can use GpuMat::copyTo with a wrapper.

                cv::cuda::GpuMat targetWrapper(mInputH, mInputW, CV_32FC1, targetPtr);
                channels[c].copyTo(targetWrapper);
            }
        }

        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        mCalibrationCache.clear();
        std::ifstream input(mCalibTableName, std::ios::binary);
        input >> std::noskipws;
        if (input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream output(mCalibTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    int mBatchSize, mInputW, mInputH;
    size_t mInputCount;
    int mCursor = 0;
    std::string mImgDir;
    std::string mCalibTableName;
    std::vector<std::string> mFilePaths;
    void* mDeviceInput{ nullptr };
    std::vector<char> mCalibrationCache;
};