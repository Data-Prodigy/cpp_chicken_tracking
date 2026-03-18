#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <Eigen/Cholesky>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// OpenCV & CUDA Modules
#include <opencv2/cudawarping.hpp>  // For resize
#include <opencv2/cudaarithm.hpp>   // For split/copyMakeBorder
#include <opencv2/cudaimgproc.hpp>  // For cvtColor
#include <opencv2/core/cuda.hpp> 
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudafilters.hpp>
#include "Int8Calibrator.h"
#include "include/BYTETracker.h"
#include "YoloInference.h"
#include <map>
#include <deque>
#include <numeric>
#include <thread>
#include <mutex>
#include <atomic>