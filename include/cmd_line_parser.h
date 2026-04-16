//
// Created by Cyrus on 5/9/2024.
//

#ifndef CMD_LINE_PARSER_H
#define CMD_LINE_PARSER_H

#include <string>
#include <vector>

struct CommandLineArguments {
    std::string det_trt_model;
    std::string det_onnx_model;
    std::string inputPath;
    std::string rec_trt_model;
    std::string rec_onnx_model;
    std::string port;
    std::string model_dir;
    std::string label_file;
    std::string config_file;
    bool use_onnx; // Add flag for ONNX model usage
    bool use_onnx_set; // Flag to check if use_onnx was explicitly set
};

bool parseArguments(int argc, char* argv[], CommandLineArguments& arguments);

std::string getLogLevelFromEnvironment();

#endif //CMD_LINE_PARSER_H