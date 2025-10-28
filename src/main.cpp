
#include "cmd_line_parser.h"
#include "logger.h"
#include "ocr.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    CommandLineArguments arguments;
    //HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    //if (hOut != INVALID_HANDLE_VALUE)
    //    SetConsoleOutputCP(CP_UTF8);

    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel("info");
    spdlog::set_level(logLevel);
    std::unique_ptr<char> pDet = nullptr;
    if (argc == 1) {
        argc = 7;
        pDet = std::make_unique<char>(1024);
        auto start = 0;
        argv[1] = pDet.get() + start;
        auto toCopy = sizeof("--det_onnx_model");
        if (toCopy + start > 1024){
            return -1;
        }
        strcpy(pDet.get() + start, "--det_onnx_model");
        pDet.get()[start + toCopy - 1] = '\0';
        start += toCopy;

        argv[2] = pDet.get() + start;
        toCopy = sizeof("..\\..\\..\\models\\en_PP-OCRv3_det_infer.onnx");
        if (toCopy + start > 1024){
            return -1;
        }
        strcpy(pDet.get() + start, "..\\..\\..\\models\\en_PP-OCRv3_det_infer.onnx");
        pDet.get()[start + toCopy - 1] = '\0';
        start += toCopy;

        argv[3] = pDet.get() + start;
        toCopy = sizeof("--input");
        if (toCopy + start > 1024){
            return -1;
        }
        strcpy(pDet.get() + start, "--input");
        pDet.get()[start + toCopy - 1] = '\0';
        start += toCopy;

        argv[4] = pDet.get() + start;
        toCopy = sizeof("I:\\img\\cap\\cap-shadower1.png");
        if (toCopy + start > 1024){
            return -1;
        }
        strcpy(pDet.get() + start, "I:\\img\\cap\\cap-shadower1.png");
        pDet.get()[start + toCopy - 1] = '\0';
        start += toCopy;

        argv[5] = pDet.get() + start;
        toCopy = sizeof("--rec_onnx_model");
        if (toCopy + start > 1024){
            return -1;
        }
        strcpy(pDet.get() + start, "--rec_onnx_model");
        pDet.get()[start + toCopy - 1] = '\0';
        start += toCopy;

        argv[6] = pDet.get() + start;
        toCopy = sizeof("..\\..\\..\\models\\en_PP-OCRv4_rec_infer.onnx");
        if (toCopy + start > 1024){
            return -1;
        }
        strcpy(pDet.get() + start, "..\\..\\..\\models\\en_PP-OCRv4_rec_infer.onnx");
        pDet.get()[start + toCopy - 1] = '\0';
        //start += toCopy;
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

    //cpuImg.resize();
    //cv::resize(cpuImg, cpuImg, cv::Size(256, 100));

    // In the following section we populate the input vectors to later pass for
    // inference
    
    ocr instance;    
    std::vector<double> ocr_times;

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
