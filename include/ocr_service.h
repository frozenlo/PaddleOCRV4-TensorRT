#pragma once

#include "httplib.h"
#include "ocr.h"
#include <memory>

class OCRService {
public:
    OCRService(const std::string& det_model_path, const std::string& rec_model_path, const std::string& label_path, int port = 8080, bool use_onnx = false);
    ~OCRService();
    
    void start();
    void stop();
    
private:
    void setupRoutes();
    
    // HTTP handlers
    void handleOCR(httplib::Request& req, httplib::Response& res);
    void handleHealth(httplib::Request& req, httplib::Response& res);
    
    int port_;
    std::unique_ptr<httplib::Server> server_;
    std::unique_ptr<ocr> ocr_instance_;
    
    // Model paths
    std::string det_model_path_;
    std::string rec_model_path_;
    std::string label_path_;
    bool use_onnx_;
};