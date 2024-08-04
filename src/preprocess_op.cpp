#pragma once
#include "preprocess_op.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

namespace OCR {

void Permute::Run(cv::cuda::GpuMat &im) {

    cv::cuda::GpuMat tmp = im.clone();
    size_t width = im.rows * im.cols;
    std::vector<cv::cuda::GpuMat> input_channels{cv::cuda::GpuMat(im.rows, im.cols, CV_32FC1, &(im.ptr()[0])),
                                                 cv::cuda::GpuMat(im.rows, im.cols, CV_32FC1, &(im.ptr()[width * 4])),
                                                 cv::cuda::GpuMat(im.rows, im.cols, CV_32FC1, &(im.ptr()[width * 8]))};
    cv::cuda::split(tmp, input_channels); // HWC -> CHW
}

void PermuteBatch::Run(const std::vector<cv::cuda::GpuMat> imgs, cv::cuda::GpuMat &dest) {
    dest = cv::cuda::GpuMat(imgs.size(), imgs[0].rows * imgs[0].cols * 3, CV_32FC1);
    for (int j = 0; j < imgs.size(); j++) {
        int rh = imgs[j].rows;
        int rw = imgs[j].cols;
        size_t width = rh * rw;
        size_t start = width * 4 * 3 * j;
        std::vector<cv::cuda::GpuMat> input_channels{cv::cuda::GpuMat(rh, rw, CV_32FC1, &(dest.ptr()[start])),
                                                     cv::cuda::GpuMat(rh, rw, CV_32FC1, &(dest.ptr()[start + width * 4])),
                                                     cv::cuda::GpuMat(rh, rw, CV_32FC1, &(dest.ptr()[start + width * 8]))};

        cv::cuda::split(imgs[j], input_channels);
    }
}

void Normalize::Run(cv::cuda::GpuMat &im, const std::array<float, 3> &mean, const std::array<float, 3> &scale, const bool is_scale) {
    im.convertTo(im, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(im, cv::Scalar(mean[0], mean[1], mean[2]), im, cv::noArray(), -1);
    cv::cuda::multiply(im, cv::Scalar(scale[0], scale[1], scale[2]), im, 1, -1);
    // Apply scaling and mean subtraction
}



void ResizeImgType0::Run(const cv::cuda::GpuMat &img, cv::cuda::GpuMat &resize_img, int max_size_len, float &ratio_h, float &ratio_w) {

    int resize_w;
    int resize_h;

    get_resize_ratio(img.cols, img.rows, max_size_len, resize_h, resize_w);

    cv::cuda::resize(img, resize_img, cv::Size(resize_w, resize_h));
    ratio_h = float(resize_h) / float(img.rows);
    ratio_w = float(resize_w) / float(img.cols);
}

void ResizeImgType0_Cpu::Run(const cv::Mat &img, cv::Mat &resize_img, int max_size_len, float &ratio_h, float &ratio_w) {

    int resize_w;
    int resize_h;

    get_resize_ratio(img.cols, img.rows, max_size_len, resize_h, resize_w);

    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    ratio_h = float(resize_h) / float(img.rows);
    ratio_w = float(resize_w) / float(img.cols);
}

void CrnnResizeImg::Run(const cv::cuda::GpuMat &img, cv::cuda::GpuMat &resize_img, float wh_ratio,
                        const std::array<int, 3> &rec_image_shape) {
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    imgW = int(48 * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::cuda::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f, cv::INTER_LINEAR);
    cv::cuda::copyMakeBorder(resize_img, resize_img, 0, 0, 0, int(imgW - resize_img.cols), cv::BORDER_CONSTANT, {127, 127, 127});
}

void ClsResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img, const std::vector<int> &rec_image_shape) {
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;
    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));

    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f, cv::INTER_LINEAR);
    if (resize_w < imgW) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
}

} // namespace OCR
