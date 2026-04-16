#include "ocr_service.h"
#include "spdlog/spdlog.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

OCRService::OCRService(const std::string& det_model_path, const std::string& rec_model_path, const std::string& label_path, int port, bool use_onnx) 
    : det_model_path_(det_model_path), rec_model_path_(rec_model_path), label_path_(label_path), use_onnx_(use_onnx), port_(port) {
    server_ = std::make_unique<httplib::Server>();
    ocr_instance_ = std::make_unique<ocr>(label_path);
    
    // Initialize OCR models
    if (use_onnx_) {
        ocr_instance_->Model_Init("", det_model_path_,
                                 "", rec_model_path_);
    } else {
        ocr_instance_->Model_Init(det_model_path_, det_model_path_, 
                                 rec_model_path_, rec_model_path_);
    }
    
    setupRoutes();
}

OCRService::~OCRService() {
    stop();
}

void OCRService::start() {
    spdlog::info("Starting OCR service on port {}", port_);
    
    server_->listen("0.0.0.0", port_);
}

void OCRService::stop() {
    spdlog::info("Stopping OCR service");
    server_->stop();
}

void OCRService::setupRoutes() {
    server_->Post("/ocr", [this](const httplib::Request& req, httplib::Response& res) {
        handleOCR(const_cast<httplib::Request&>(req), res);
    });
    
    server_->Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handleHealth(const_cast<httplib::Request&>(req), res);
    });
}

void OCRService::handleOCR(httplib::Request& req, httplib::Response& res) {
    try {
        // Check if the request has a file with key "image"
        if (!req.form.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\": \"No image file provided\"}", "application/json");
            return;
        }
        
        // Get the file
        auto file = req.form.get_file("file", 0);
        
        // Save image to memory
        std::vector<uchar> data(file.content.begin(), file.content.end());
        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        
        if (img.empty()) {
            res.status = 400;
            res.set_content("{\"error\": \"Invalid image data\"}", "application/json");
            return;
        }
        
        // Process OCR
        std::vector<double> ocr_times;
        auto results = ocr_instance_->Model_Infer(img, ocr_times);
        
        // Format results as JSON
        json response;
        response["status"] = "success";
        response["results"] = json::array();
        
        for (const auto& result : results) {
            json text_result;
            text_result["text"] = result.text;
            text_result["confidence"] = result.confidence;
            
            // Add bounding box information
            json box_array = json::array();
            for (const auto& point : result.box) {
                json point_array = json::array();
                for (const auto& coord : point) {
                    point_array.push_back(coord);
                }
                box_array.push_back(point_array);
            }
            text_result["box"] = box_array;
            
            response["results"].push_back(text_result);
        }
        
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        spdlog::error("Error processing OCR request: {}", e.what());
        res.status = 500;
        json error_response;
        error_response["error"] = "Internal server error";
        error_response["message"] = e.what();
        res.set_content(error_response.dump(), "application/json");
    }
}

void OCRService::handleHealth(httplib::Request& req, httplib::Response& res) {
    json response;
    response["status"] = "healthy";
    response["service"] = "PaddleOCR TensorRT Service";
    response["use_onnx"] = use_onnx_;
    res.set_content(response.dump(), "application/json");
}