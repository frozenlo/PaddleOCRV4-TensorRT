#include "config_loader.h"
#include <fstream>
#include <iostream>
#include <filesystem>

nlohmann::json ConfigLoader::loadConfig(const CommandLineArguments& args) {
    // Load default config
    nlohmann::json config = loadFromFile("config.json");
    
    // Override with command line arguments if provided
    if (!args.model_dir.empty()) {
        config["model_dir"] = args.model_dir;
    }
    
    if (!args.det_trt_model.empty()) {
        config["det_trt_model"] = args.det_trt_model;
    }
    
    if (!args.det_onnx_model.empty()) {
        config["det_onnx_model"] = args.det_onnx_model;
    }
    
    if (!args.rec_trt_model.empty()) {
        config["rec_trt_model"] = args.rec_trt_model;
    }
    
    if (!args.rec_onnx_model.empty()) {
        config["rec_onnx_model"] = args.rec_onnx_model;
    }
    
    if (!args.label_file.empty()) {
        config["label_file"] = args.label_file;
    }
    
    if (!args.port.empty()) {
        config["port"] = args.port;
    }
    
    // Handle use_onnx flag
    if (args.use_onnx_set) {
        config["use_onnx"] = args.use_onnx;
    }
    
    // Expand variables in paths
    if (config.contains("det_trt_model")) {
        config["det_trt_model"] = expandVariables(config["det_trt_model"], config);
    }
    
    if (config.contains("det_onnx_model")) {
        config["det_onnx_model"] = expandVariables(config["det_onnx_model"], config);
    }
    
    if (config.contains("rec_trt_model")) {
        config["rec_trt_model"] = expandVariables(config["rec_trt_model"], config);
    }
    
    if (config.contains("rec_onnx_model")) {
        config["rec_onnx_model"] = expandVariables(config["rec_onnx_model"], config);
    }
    
    if (config.contains("label_file")) {
        config["label_file"] = expandVariables(config["label_file"], config);
    }
    
    return config;
}

nlohmann::json ConfigLoader::loadFromFile(const std::string& configFile) {
    nlohmann::json config;
    
    // Check if config file exists
    if (!std::filesystem::exists(configFile)) {
        std::cerr << "Config file not found: " << configFile << ", using defaults" << std::endl;
        // Return empty json object with default structure
        return config;
    }
    
    try {
        std::ifstream file(configFile);
        if (file.is_open()) {
            file >> config;
            file.close();
        } else {
            std::cerr << "Failed to open config file: " << configFile << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config file: " << e.what() << std::endl;
    }
    
    return config;
}

void ConfigLoader::mergeConfigs(nlohmann::json& base, const nlohmann::json& overrides) {
    for (auto& [key, value] : overrides.items()) {
        base[key] = value;
    }
}

std::string ConfigLoader::expandVariables(const std::string& value, const nlohmann::json& config) {
    std::string result = value;
    
    // Replace {model_dir} with actual model_dir value
    if (config.contains("model_dir")) {
        std::string modelDir = config["model_dir"];
        size_t pos = result.find("{model_dir}");
        while (pos != std::string::npos) {
            result.replace(pos, 11, modelDir);
            pos = result.find("{model_dir}", pos + modelDir.length());
        }
    }
    
    return result;
}