#pragma once
#include "opencv2/core.hpp"

namespace OCR {


inline void get_resize_ratio(int w, int h, int max_size_len, int min_size_len, int &resize_h, int &resize_w) {

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) { // If the longest side exceeds the maximum size, scale down proportionally
        if (h > w) {
            ratio = float(max_size_len) / float(h);
        } else {
            ratio = float(max_size_len) / float(w);
        }
    }

    resize_h = int(float(h) * ratio);
    resize_w = int(float(w) * ratio);

    // Use min_size_len instead of hard-coded values
    resize_h = std::max(int(round(float(resize_h) / min_size_len) * min_size_len), min_size_len);
    resize_w = std::max(int(round(float(resize_w) / min_size_len) * min_size_len), min_size_len);
}


class Normalize {
public:
   void Run(cv::cuda::GpuMat& im, const std::array<float,3> &mean,
                   const std::array<float,3> &scale, const bool is_scale = true);
};


// RGB -> CHW
class Permute {
public:
   void Run(cv::cuda::GpuMat& im);
};

class PermuteBatch {
public:
     void Run(const std::vector<cv::cuda::GpuMat> imgs, cv::cuda::GpuMat &dest);
};
    
class ResizeImgType0 {
public:
   void Run(const cv::cuda::GpuMat &img, cv::cuda::GpuMat &resize_img, int max_size_len, int min_size_len,
                   float &ratio_h, float &ratio_w);
};

class ResizeImgType0_Cpu {
public:
     void Run(const cv::Mat &img, cv::Mat &resize_img, int max_size_len, int min_size_len, float &ratio_h, float &ratio_w);
};

class CrnnResizeImg {
public:
     void Run(const cv::cuda::GpuMat &img, cv::cuda::GpuMat &resize_img, int base_size, float wh_ratio, const std::array<int,3> &rec_image_shape = {3, 48, 320});
};

class ClsResizeImg {
public:
   void Run(const cv::Mat &img, cv::Mat &resize_img,
                   const std::vector<int> &rec_image_shape = {3, 48, 192});
};

} // namespace OCR