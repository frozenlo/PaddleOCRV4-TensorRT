#pragma once

#include "engine.h"
#include "postprocess_op.h"
#include "preprocess_op.h"


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
    class TextDetect: public Engine<float>{
    public:
        TextDetect() : Engine(){};
        void Model_Infer(const cv::cuda::GpuMat& img, vector<vector<vector<int>>> &boxes, vector<double> &times);
        void Model_Init(std::string_view det_engine_path, std::string_view det_onnx_path);
        uint32_t getMaxOutputLength(nvinfer1::Dims tensorShape) const override;

    private:
        //config
    
        //task
        double det_db_thresh_ = 0.3;
        double det_db_box_thresh_ = 0.5;
        double det_db_unclip_ratio_ = 2.0;
        bool use_polygon_score_ = false;

        // input/output layer 
        const char *INPUT_BLOB_NAME = "x";
        const char *OUTPUT_BLOB_NAME = "save_infer_model/scale_0.tmp_1";

        // input image
        int max_side_len_ = 960;
        ResizeImgType0 resize_op_;
        //ResizeImgType0_Cpu resize_op_cpu;

        std::array<float, 3> mean_ = {0.485f, 0.456f, 0.406f};
        std::array<float, 3> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
        Normalize normalize_op_;

        Permute permute_op_;

        // output result
        PostProcessor post_processor_;
  
    };

}// namespace OCR

