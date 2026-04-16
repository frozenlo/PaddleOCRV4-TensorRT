#include "cmd_line_parser.h"
#include "config_loader.h"
#include "logger.h"
#include "ocr.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    CommandLineArguments arguments;
    
    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel("info");
    spdlog::set_level(logLevel);
    std::unique_ptr<char[]> pDetBuffer = nullptr;

    // for save default arguments
    std::vector<char*> tempArgvList;

    //set default arguments when there is no input. And can also show you how to use this.
    if (argc == 1) {
        const size_t BUFFER_SIZE = 1024;
        pDetBuffer = std::make_unique<char[]>(BUFFER_SIZE);

        size_t currentOffset = 0; 

        auto addArg = [&](const char* str) -> bool {
            size_t len = strlen(str) + 1; 

            if (currentOffset + len > BUFFER_SIZE) {
                std::cerr << "Error: Argument buffer overflow!" << std::endl;
                return false;
            }

            char* dest = pDetBuffer.get() + currentOffset;
            strcpy(dest, str);          
            tempArgvList.push_back(dest);
            currentOffset += len;
            return true;
            };

        addArg(argv[0]);

        if (!addArg("--det_onnx_model")) return -1;

        if (!addArg("../../../models/det_model.onnx")) return -1;

        if (!addArg("--input")) return -1;

        if (!addArg("I:\\img\\cap\\cap-shadower1.png")) return -1;

        if (!addArg("--rec_onnx_model")) return -1;

        if (!addArg("../../../models/rec_model.onnx")) return -1;
        argc = static_cast<int>(tempArgvList.size());
        argv = tempArgvList.data();
    }

    // Parse the command line arguments
    if (!parseArguments(argc, argv, arguments)) {
        return -1;
    }

    // Load configuration
    nlohmann::json config = ConfigLoader::loadConfig(arguments);

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

    std::string basePath = "..\\..\\..\\models\\";
    
    std::string label_file = config.value("label_file", basePath + "ppocr_keys_v1.txt");
    ocr instance(label_file);    
    std::vector<double> ocr_times;

    // Determine whether to use ONNX or TRT models
    bool use_onnx = config.value("use_onnx", false);
    
    std::string det_model, rec_model;
    if (use_onnx) {
        det_model = config.value("det_onnx_model", basePath + "det_model.onnx");
        rec_model = config.value("rec_onnx_model", basePath + "rec_model.onnx");
        spdlog::info("Using ONNX models: det={}, rec={}", det_model, rec_model);
    } else {
        det_model = config.value("det_trt_model", basePath + "det_model.trt");
        rec_model = config.value("rec_trt_model", basePath + "rec_model.trt");
        spdlog::info("Using TRT models: det={}, rec={}", det_model, rec_model);
    }
    
    instance.Model_Init(det_model, det_model, rec_model, rec_model);
    auto results = instance.Model_Infer(cpuImg, ocr_times);
    
    // Print results with bounding box information
    for (const auto& result : results) {
        std::cout << "Text: \"" << result.text << "\", Confidence: " << result.confidence << std::endl;
        
        std::cout << "Bounding box: [";
        for (size_t i = 0; i < result.box.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "[";
            for (size_t j = 0; j < result.box[i].size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << result.box[i][j];
            }
            std::cout << "]";
        }
        std::cout << "]" << std::endl;
    }
    
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
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