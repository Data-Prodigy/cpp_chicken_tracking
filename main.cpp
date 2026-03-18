#include "pch.h"

using namespace nvinfer1;
using namespace nvonnxparser;

class Logger : public ILogger { //inherits from ILOGGER & overrides log method
    void log(Severity severity, const char* msg) noexcept override {
        //suppress info messages... only print warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << " [TensorRT] "<< msg << '\n';
        }
    }
} logger;


// --CREATING THE BUILDER -- the factory that creates everything else
void buildEngineFile(const std::string& onnxPath, const std::string& enginePath) {
    IBuilder* builder = createInferBuilder(logger);
    if (!builder) {
        std::cerr << "ERROR: Failed to Create Builder!!" << '\n';
        return;
    }

    //Creating the Network with explicit Batch Flag
    INetworkDefinition* network = builder->createNetworkV2(0);
    if (!network) {
        std::cerr << "ERROR: Failed to Create Network!!" << '\n';
        return;
    }

    //Creating the ONNX Parser
    IParser* parser = createParser(*network, logger);
    if (!parser) {
        std::cerr << "ERROR: Failed to Create Parser!!" << '\n';
        return;
    }

    //Parsing the file -- this reads my onnx model and fills the 'network' structure
    std::cout << "Parsing ONNX File: " << onnxPath << "..." << '\n';
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "ERROR: Failed to Parse ONNX FILE!!" << '\n';
        return;
    }
    std::cout << "SUCCESSFULLY PARSED ONNX FILE :)" << '\n';

    IBuilderConfig* config = builder->createBuilderConfig();
    //Allocating memory for the build process --we give 1GB (1ULL << 30) 
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

    
    if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);
    
    // Check for INT8 Support
    if (builder->platformHasFastInt8()) {
        std::cout << "INT8 Support Detected. Enabling INT8 Calibration.\n";
        config->setFlag(BuilderFlag::kINT8);
        std::cout << "[DEBUG] 2. About to create Int8Calibrator (Constructor)...\n";

        // 1. Prepare Calibration Data
        // Create a folder "calib_images" and put ~100 jpgs from your dataset there.
        Int8Calibrator *calibrator = new Int8Calibrator(1, 640, 640, "C:/dev/cpp_chicken_tracking/x64/Release/calib_images", "calibration.cache");
        std::cout << "[DEBUG] 3. Constructor finished successfully.\n";

        // 2. Attach Calibrator
        config->setInt8Calibrator(calibrator);
        std::cout << "[DEBUG] 4. Calibrator attached to config.\n";
    }
 

    //Building the serialized network -- the engine 'blob'
    std::cout << "Building TensorRT Engine (may take some minutes)" << '\n';
    IHostMemory* serializedmodel = builder->buildSerializedNetwork(*network, *config);

    if (!serializedmodel) {
        std::cerr << "ERROR: Failed to Build Engine!!" << '\n';
        return;
    }
    std::cout << "ENGINE BUILT SUCCESSFULLY, YAYYYY!!!" << '\n';

    //SAVING ENGINE TO DISK (SERIALIZING) -- treating engine like a single binary blob of data
    std::ofstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "ERROR: Could Not Open Output File!!" << enginePath << '\n';
        return;
    }

    //writing the data pointer to the file
    engineFile.write(static_cast<const char*>(serializedmodel->data()), serializedmodel->size());
    engineFile.close();
    std::cout << "Engine Saved to :" << enginePath << '\n';

    //LAST STEP -- THE CLEANUP
    //DELETING MY TENSORrt objects
    delete parser;
    delete network;
    delete config;
    delete builder;
    delete serializedmodel;
}

// Helper: Intersection over Union (IoU)
float getIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;

    if (area1 + area2 - intersectionArea == 0) return 0.0f;
    return (float)intersectionArea / (area1 + area2 - intersectionArea);
}

class AsyncCamera {
public:
    // Constructor 1: For USB / Virtual Webcams (DroidCam PC Client)
    AsyncCamera(int cameraIndex, int width, int height, int fps) {
        // Use MSMF (Media Foundation) instead of DSHOW. It is the modern Windows API 
        // and handles virtual webcams (like DroidCam/OBS) without crashing.
        cap.open(cameraIndex, cv::CAP_MSMF);

        // Fallback to ANY if MSMF fails
        if (!cap.isOpened()) {
            std::cout << "MSMF failed, trying default backend...\n";
            cap.open(cameraIndex, cv::CAP_ANY);
        }

        init(width, height, fps);
    }

    // Constructor 2: For Wi-Fi / IP Cameras (DroidCam App URL)
    AsyncCamera(const std::string& streamUrl, int width, int height, int fps) {
        // Use FFMPEG for network streams
        cap.open(streamUrl, cv::CAP_FFMPEG);

        // Force OpenCV to only keep the 1 newest frame in the network buffer
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        init(width, height, fps);
    }

    ~AsyncCamera() {
        isRunning = false;
        if (workerThread.joinable()) {
            workerThread.join();
        }
        cap.release();
    }

    bool isOpened() { return cap.isOpened(); }

    bool read(cv::Mat& outputFrame) {
        std::lock_guard<std::mutex> lock(frameMutex);
        if (latestFrame.empty()) return false;
        latestFrame.copyTo(outputFrame);
        return true;
    }

private:
    cv::VideoCapture cap;
    cv::Mat latestFrame;
    std::mutex frameMutex;
    std::atomic<bool> isRunning{ false };
    std::thread workerThread;

    void init(int width, int height, int fps) {
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            cap.set(cv::CAP_PROP_FPS, fps);

            isRunning = true;
            workerThread = std::thread(&AsyncCamera::update, this);
            std::cout << "Camera initialized successfully on background thread.\n";
        }
        else {
            std::cerr << "Error: Camera failed to open.\n";
        }
    }

    void update() {
        while (isRunning) {
            cv::Mat tempFrame;
            // This grabs frames as fast as the network/USB can push them!
            if (cap.read(tempFrame) && !tempFrame.empty()) {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = tempFrame;
            }
        }
    }
};

int main() {
    AsyncCamera asyncCap(1, 1280, 720, 50);
    if (!asyncCap.isOpened()) {
        std::cerr << "Exiting: Camera not found." << std::endl;
        return -1;
    }

    YoloInference yolo("yolov11s.engine");
    BYTETracker tracker(250, 0.25f, 0.1f, 0.30f, 0.80f);
    cv::Mat frame;

    // --- PRODUCTION HEALTH REGISTRY ---
    struct HealthState {
        float sickConviction = 0.0f;   // Range: 0.0 (Perfectly Healthy) to 1.0 (Absolutely Sick)
        bool isPermanentlyRed = false; // The Latch
        int framesTracked = 0;         // How long have we known this chicken?
    };
    std::map<int, HealthState> chickenHealthDb;

    // --- TUNING PARAMETERS ---
    const float CONVICTION_LATCH_THRESHOLD = 0.75f; // Human-level sureness required to lock RED
    const float LEARNING_RATE = 0.15f;              // How fast the system changes its mind (15% per frame)
    const float MIN_CONFIDENCE_TO_CARE = 0.40f;     // Ignore blurry/uncertain guesses entirely
    // --- TEMPORAL MASK MEMORY ---
    std::map<int, cv::Mat> maskHistory;
    const float MASK_LEARNING_RATE = 0.35f; // 35% New Frame, 65% Old Frame (Higher = more responsive, Lower = smoother)
   
    struct TrackMaskState {
        cv::Mat mask;
    };

    std::map<int, TrackMaskState> trackMasks;
    while (true) {
        if (!asyncCap.read(frame)) {
            // If the camera thread hasn't grabbed a frame yet, just loop and wait a millisecond
            continue;
        }
        if (frame.empty()) {
            std::cout << "Video ended. Looping...\n";
            chickenHealthDb.clear();
            maskHistory.clear();
            continue;
        }

        cv::Mat smallFrame;
        cv::resize(frame, smallFrame, cv::Size(1280, 720));

        // 1. Run Inference
        std::vector<Detection> detections = yolo.runInference(smallFrame);

        // 2. Prepare Tracker Inputs
        std::vector<Object> trackObjects;
        for (const auto& det : detections) {
            Object obj;
            obj.rect = cv::Rect_<float>(det.box.x, det.box.y, det.box.width, det.box.height);
            obj.label = det.class_id;
            obj.prob = det.confidence;
            trackObjects.push_back(obj);
        }

        // 3. Update Tracker
        std::vector<STrack> lost_stracks;
        std::vector<STrack> output_stracks;
        tracker.update(trackObjects, lost_stracks, output_stracks);

        // 1. Initialize frame counters
        int healthyCount = 0;
        int unhealthyCount = 0;

        // --- DRAW RESULTS ---
        for (const auto& t : output_stracks) {

            std::vector<float> tlwh = t.tlwh;
            cv::Rect trackRect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);

            HealthState& state = chickenHealthDb[t.track_id];

            int bestIdx = -1;
            float bestScore = -1.0f;

            // --- MATCH TRACK TO DETECTION ---
            for (size_t i = 0; i < detections.size(); ++i) {

                float iou = getIoU(trackRect, detections[i].box);

                float centerDist = cv::norm(
                    (trackRect.tl() + trackRect.br()) * 0.5 -
                    (detections[i].box.tl() + detections[i].box.br()) * 0.5
                );

                float score = iou - 0.001f * centerDist;

                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = i;
                }
            }

            // --- HEALTH UPDATE ---
            if (bestIdx != -1) {

                const auto& det = detections[bestIdx];
                state.framesTracked++;

                float frameScore = 0.0f;

                if (det.confidence > MIN_CONFIDENCE_TO_CARE) {
                    frameScore = (det.class_id == 1) ? det.confidence : 0.0f;
                }
                else {
                    frameScore = state.sickConviction;
                }

                state.sickConviction =
                    state.sickConviction * (1.0f - LEARNING_RATE) +
                    frameScore * LEARNING_RATE;

                if (!state.isPermanentlyRed &&
                    state.sickConviction >= CONVICTION_LATCH_THRESHOLD &&
                    state.framesTracked > 15) {

                    state.isPermanentlyRed = true;

                    std::cout << "[ALERT] Chicken ID "
                        << t.track_id
                        << " locked UNHEALTHY\n";

                    int alert_id = t.track_id;

                    std::thread([alert_id]() {
                        std::string scriptPath =
                            "C:/dev/cpp_chicken_tracking/cpp_chicken_tracking/send_alerts.py";
                        std::string cmd =
                            "python " + scriptPath + " " + std::to_string(alert_id);
                        system(cmd.c_str());

                        }).detach();
                }

                // STORE MASK FOR THIS TRACK
                trackMasks[t.track_id].mask = det.boxMask.clone();
            }

            // --- MASK SOURCE ---
            cv::Mat mask;
            if (bestIdx != -1) {
                mask = detections[bestIdx].boxMask.clone();
            }
            else if (trackMasks.find(t.track_id) != trackMasks.end()) {
               mask = trackMasks[t.track_id].mask.clone();
            }

            // --- DRAW MASK ---
            if (!mask.empty()) {
                cv::Rect destROI = trackRect &
                    cv::Rect(0, 0, smallFrame.cols, smallFrame.rows);

                if (destROI.area() > 0) {
                    cv::Mat fullMask;
                    cv::resize(mask, fullMask,
                        destROI.size(),
                        0, 0,
                        cv::INTER_NEAREST);

                    // TEMPORAL SMOOTHING
                    if (maskHistory.find(t.track_id) != maskHistory.end()) {

                        cv::Mat oldMask = maskHistory[t.track_id];

                        if (oldMask.size() != fullMask.size()) {
                            cv::resize(oldMask, oldMask,
                                fullMask.size());
                        }

                        cv::addWeighted(
                            oldMask,
                            1.0f - MASK_LEARNING_RATE,
                            fullMask,
                            MASK_LEARNING_RATE,
                            0.0,
                            fullMask
                        );
                    }

                    maskHistory[t.track_id] = fullMask.clone();

                    cv::Mat binaryMask;
                    cv::threshold(fullMask,
                        binaryMask,
                        0.5,
                        255,
                        cv::THRESH_BINARY);

                    binaryMask.convertTo(binaryMask, CV_8U);

                    cv::Scalar maskColor =
                        state.isPermanentlyRed ?
                        cv::Scalar(0, 0, 255) :
                        cv::Scalar(0, 255, 0);

                    cv::Mat roi = smallFrame(destROI);

                    cv::Mat colorLayer(
                        roi.size(),
                        CV_8UC3,
                        maskColor
                    );

                    cv::Mat blended;

                    cv::addWeighted(
                        roi,
                        0.5,
                        colorLayer,
                        0.5,
                        0,
                        blended
                    );

                    blended.copyTo(roi, binaryMask);
                }
            }

            // --- COUNT ---
            if (state.isPermanentlyRed)
                unhealthyCount++;
            else
                healthyCount++;

            // --- DRAW BOX ---
            cv::Scalar boxColor =
                state.isPermanentlyRed ?
                cv::Scalar(0, 0, 255) :
                cv::Scalar(0, 255, 0);

            //cv::rectangle(smallFrame, trackRect, boxColor,2);

            std::string label =
                "ID: " + std::to_string(t.track_id);

            // 2. Calculate the size of the text box
            int baseLine;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);

            // 3. Define the text position
            cv::Point textOrg(trackRect.x + 5, trackRect.y + 20);

            // 4. Draw a filled background box 
            // We add a small 5px padding for a cleaner look
            cv::rectangle(
                smallFrame,
                textOrg + cv::Point(0, baseLine),                         // Bottom-left
                textOrg + cv::Point(textSize.width, -textSize.height),    // Top-right
                cv::Scalar(30, 30, 30),                                   // Dark Gray/Black box
                cv::FILLED
            );

            // 5. Now draw the text on top of the box
            cv::putText(
                smallFrame,
                label,
                textOrg,
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                boxColor, // This will be your light green or light red
                2
            );
        }

        // FPS Counter
        static auto lastTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float fps = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();
        lastTime = currentTime;

        // --- 1. HUD BACKGROUND (Semi-Transparent Box) ---
        // Define the dimensions of the box (x, y, width, height)
        // Make it wide enough to fit the text, and tall enough to cover the 3 lines
        cv::Rect hudRect(5, 5, 190, 90);

        // Safety clamp to ensure the box doesn't go off-screen
        hudRect &= cv::Rect(0, 0, smallFrame.cols, smallFrame.rows);

        if (hudRect.area() > 0) {
            // Extract the background pixels from the video frame
            cv::Mat roi = smallFrame(hudRect);

            // Create a solid white rectangle of the exact same size
            cv::Mat whiteBox(roi.size(), CV_8UC3, cv::Scalar(255, 255, 255));

            // Blend them! 
            // 0.35 = 35% White Box (Opacity) | 0.65 = 65% Original Background
            double alpha = 0.65;
            cv::addWeighted(whiteBox, alpha, roi, 1.0 - alpha, 0.0, roi);
        }

        cv::putText(smallFrame, "FPS: " + std::to_string((int)fps), cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(167,123,0), 2);
        std::string healthyText = "Healthy: " + std::to_string(healthyCount);
        cv::putText(smallFrame, healthyText, cv::Point(10, 60),cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(148,148,13), 2);

        // Draw Unhealthy Count (Red) right below it
        std::string unhealthyText = "Unhealthy: " + std::to_string(unhealthyCount);
        cv::putText(smallFrame, unhealthyText, cv::Point(10, 80),cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(51,52,202), 2);

        cv::imshow("Chicken Tracker", smallFrame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}