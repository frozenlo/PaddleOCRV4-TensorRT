#include "cmd_line_parser.h"
#include <iostream>
#include <cstring>

bool parseArguments(int argc, char* argv[], CommandLineArguments& arguments) {
    // Initialize boolean flags
    arguments.use_onnx = false;
    arguments.use_onnx_set = false;
    
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--det_trt_model") == 0 || strcmp(arg, "-dtm") == 0) {
            if (i + 1 < argc) {
                arguments.det_trt_model = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--det_onnx_model") == 0 || strcmp(arg, "-dom") == 0) {
            if (i + 1 < argc) {
                arguments.det_onnx_model = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--input") == 0 || strcmp(arg, "-i") == 0) {
            if (i + 1 < argc) {
                arguments.inputPath = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--rec_trt_model") == 0 || strcmp(arg, "-rtm") == 0) {
            if (i + 1 < argc) {
                arguments.rec_trt_model = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--rec_onnx_model") == 0 || strcmp(arg, "-rom") == 0) {
            if (i + 1 < argc) {
                arguments.rec_onnx_model = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--port") == 0 || strcmp(arg, "-p") == 0) {
            if (i + 1 < argc) {
                arguments.port = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--model_dir") == 0 || strcmp(arg, "-md") == 0) {
            if (i + 1 < argc) {
                arguments.model_dir = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--label_file") == 0 || strcmp(arg, "-lf") == 0) {
            if (i + 1 < argc) {
                arguments.label_file = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--config") == 0 || strcmp(arg, "-c") == 0) {
            if (i + 1 < argc) {
                arguments.config_file = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for " << arg << std::endl;
                return false;
            }
        } else if (strcmp(arg, "--use_onnx") == 0) {
            arguments.use_onnx = true;
            arguments.use_onnx_set = true;
        } else if (strcmp(arg, "--use_trt") == 0) {
            arguments.use_onnx = false;
            arguments.use_onnx_set = true;
        } else {
            std::cerr << "Error: Unknown argument " << arg << std::endl;
            return false;
        }
    }

    return true;
}