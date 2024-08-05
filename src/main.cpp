
#include "cmd_line_parser.h"
#include "logger.h"
#include "ocr.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    CommandLineArguments arguments;

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel("info");
    spdlog::set_level(logLevel);

    if (argc == 1) {
        argc = 7;
        argv[1] = "--det_onnx_model";
        argv[2] = "..\\..\\..\\models\\det_model.onnx";
        argv[3] = "--input";
        argv[4] = "..\\..\\..\\inputs\\12.jpg";
        //argv[4] = "I:\\img\\cap\\a1.png";
        argv[5] = "--rec_onnx_model";
        argv[6] = "..\\..\\..\\models\\rec_model.onnx";
    }

    // Parse the command line arguments
    if (!parseArguments(argc, argv, arguments)) {
        return -1;
    }

    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;


    // Read the input image
    // TODO: You will need to read the input image required for your model
    std::string inputImage = "..\\..\\..\\inputs\\12.jpg";
    if (!arguments.inputPath.empty()) {
        inputImage = arguments.inputPath;
    }

    cv::Mat cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        const std::string msg = "Unable to read image at path: " + inputImage;
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // In the following section we populate the input vectors to later pass for
    // inference
    
    ocr instance;    
    vector<double> ocr_times;

    instance.Model_Init(arguments.det_trt_model, arguments.det_onnx_model, arguments.rec_trt_model, arguments.rec_onnx_model);
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    instance.Model_Infer(cpuImg, ocr_times);
    _CrtDumpMemoryLeaks();

    system("pause");
    /*TextRec r;
    r.Model_Init(arguments.rec_trt_model, arguments.rec_onnx_model);
    cv::Mat ii = cv::imread("i:\\test27.png");
    cv::cuda::GpuMat input;
    input.upload(ii);
    r.inferSingle(input);*/

    return 0;
}
