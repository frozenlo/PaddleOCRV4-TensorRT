#include "ocr_service.h"
#include "cmd_line_parser.h"
#include "config_loader.h"
#include "logger.h"
#include <iostream>

int main(int argc, char* argv[]) {
    CommandLineArguments arguments;
    
    std::string logLevelStr = getLogLevelFromEnvironment();
    spdlog::level::level_enum logLevel = toSpdlogLevel("info");
    spdlog::set_level(logLevel);

    // Parse the command line arguments
    if (!parseArguments(argc, argv, arguments)) {
        std::cerr << "Failed to parse command line arguments" << std::endl;
        return -1;
    }

    // Load configuration
    nlohmann::json config = ConfigLoader::loadConfig(arguments);

    try {
        std::string label_file = config.value("label_file", "../models/ppocr_keys_v1.txt");
        int port = config.value("default_port", 8080);
        
        // Override port with command line argument if provided
        if (!arguments.port.empty()) {
            port = std::stoi(arguments.port);
        }
        
        // Determine whether to use ONNX or TRT models
        bool use_onnx = config.value("use_onnx", true);
        
        std::string det_model, rec_model;
        if (use_onnx) {
            det_model = config.value("det_onnx_model", "../models/det_model.onnx");
            rec_model = config.value("rec_onnx_model", "../models/rec_model.onnx");
            spdlog::info("Using ONNX models: det={}, rec={}", det_model, rec_model);
        } else {
            det_model = config.value("det_trt_model", "../models/det_model.trt");
            rec_model = config.value("rec_trt_model", "../models/rec_model.trt");
            spdlog::info("Using TRT models: det={}, rec={}", det_model, rec_model);
        }
        
        OCRService service(det_model, rec_model, label_file, port, use_onnx);
        service.start();
    } catch (const std::exception& e) {
        std::cerr << "Error starting OCR service: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}