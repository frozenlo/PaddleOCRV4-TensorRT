#pragma once
#include "det.h"
#include "rec.h"
#include <vector>

using namespace OCR;

// Define a structure to hold both text and its bounding box
struct TextResult {
    std::string text;                  // Single string instead of vector of strings
    std::vector<std::vector<int>> box; // Bounding box coordinates
    double confidence;
};

class ocr{
public:
    ocr(const std::string& label_path);
    ocr();
    void Model_Init(std::string det_model_path, std::string det_onnx_path, std::string rec_model_path, std::string rec_onnx_path);
    std::vector<TextResult> Model_Infer(cv::Mat& inputImg, std::vector<double> & ocr_times);
    std::string TaskProcess(const std::vector<std::pair< std::vector<std::string>, double>> &result);
    std::string MultiFrameSmooth(std::string door_result, int step);
    ~ocr();
private:

    TextDetect * td = NULL;
    TextRec * tr = NULL;

    // tast
    std::vector<int> REC_RANGE_ = {400, 599};
    float REC_THR_ = 0.85;

    //MultiFrameSmooth
    int count_img_ = 0;
    std::unordered_map<std::string, int> results_;

    bool visualize_= true;
    int count_name_ = 0;
};