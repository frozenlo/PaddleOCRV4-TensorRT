#pragma once
#include "engine.h"
#include "postprocess_op.h"
#include "preprocess_op.h"
#include "utility.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

namespace OCR {

class TextRec: public Engine<float>{
public:
    TextRec() : Engine() {
        this->m_options.MIN_DIMS_ = {1, 3, 48, 10};
        this->m_options.OPT_DIMS_ = {1, 3, 48, 320};
        this->m_options.MAX_DIMS_ = {8, 3, 48, 2000};
        this->label_list_ = Utility::ReadDict(this->label_path);
        this->label_list_.insert(this->label_list_.begin(), "#"); // blank char for ctc
        this->label_list_.push_back(" ");
    };
    void Model_Init(std::string_view det_engine_path, std::string_view det_onnx_path);
    void Model_Infer(std::vector<cv::cuda::GpuMat> &img_list, std::vector<pair<vector<string>, double>> &rec_res, vector<int> &idx_map,
                     vector<double> &times);
    virtual ~TextRec();

private:
    //task
    std::vector<std::string> label_list_;
    string label_path = "../../../models/ppocr_keys_v1.txt";


    // input image
    int rec_batch_num_=6;
    CrnnResizeImg resize_op_;

    std::array<float, 3> mean_ = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    Normalize normalize_op_;

    PermuteBatch permute_op_;

    // output result
    PostProcessor post_processor_;
  
};

}// namespace OCR