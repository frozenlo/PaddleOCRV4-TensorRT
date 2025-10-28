#pragma once
#include "utility.h"
//#include <opencv2/freetype.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace OCR {

std::vector<std::string> Utility::ReadDict(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    std::vector<std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
    } else {
        std::cout << "no such label file: " << path << ", exit the program..." << std::endl;
        exit(1);
    }
    return m_vec;
}

void Utility::VisualizeBboxes(const cv::Mat &srcimg, const std::vector<std::vector<std::vector<int>>> &boxes,
                              std::vector<std::pair<std::vector<std::string>, double>> &rec_res, std::string &detimg) {

    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    // std::cout<< boxes.size() << "," << boxes[0].size() << "," << boxes[0][0].size()<<std::endl;
    // std::cout<< rec_res.size() <<std::endl;
    // for (int n = 0; n < std::min(boxes.size(), rec_res.size()); n++) {
    for (int n = 0; n < rec_res.size(); n++) {
        cv::Point rook_points[4];
        for (int m = 0; m < boxes[n].size(); m++) {
            rook_points[m] = cv::Point(int(boxes[n][m][0]), int(boxes[n][m][1]));
        }

        const cv::Point *ppt[1] = {rook_points};
        int npt[] = {4};
        cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);

        std::string res = "";
        std::vector<std::string> temp_res = rec_res[n].first;

        for (int m = 0; m < temp_res.size(); m++) {
            res += temp_res[m];
        }

        res = res + " " + std::to_string(rec_res[n].second);

        // cv::putText(img_vis, res, rook_points[0], cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
        //std::cout << "res:" << res << std::endl;
        spdlog::info("res : {}", res);
        
        //cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();
        //ft2->loadFontData("c:\\windows\\Fonts\\simsun.ttc", 0);
        //ft2->putText(img_vis, res, rook_points[0], 30, cv::Scalar(0, 0, 255), 1, 8, false);
    }
    cv::imshow("ocr res", img_vis);
    //cv::imwrite("result.jpg", img_vis);
    cv::waitKey(0);

    // cv::imwrite(detimg, img_vis);
    // std::cout << "The detection visualized image saved in ./ocr_vis.png" << std::endl;
}

// list all files under a directory
void Utility::GetAllFiles(const char *dir_name, std::vector<std::string> &all_inputs) {
    if (NULL == dir_name) {
        std::cout << " dir_name is null ! " << std::endl;
        return;
    }
    const auto dir = fs::path(dir_name);

    if (!fs::is_directory(dir)) {
        std::cout << "dir_name is not a valid directory !" << std::endl;
        all_inputs.push_back(dir_name);
        return;
    } else {
        for (const auto &entry : fs::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                all_inputs.push_back(entry.path().string());
            }
        }
    }
}

cv::cuda::GpuMat Utility::GetRotateCropImage(const cv::cuda::GpuMat &srcImage, const std::vector<std::vector<int>> &box) {
    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    int img_crop_width = int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
    int img_crop_height = int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(box[0][0], box[0][1]);
    pointsf[1] = cv::Point2f(box[1][0], box[1][1]);
    pointsf[2] = cv::Point2f(box[2][0], box[2][1]);
    pointsf[3] = cv::Point2f(box[3][0], box[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::cuda::GpuMat dst_img;
    cv::cuda::warpPerspective(srcImage, dst_img, M, cv::Size(img_crop_width, img_crop_height), cv::BORDER_REPLICATE);

    cv::cuda::GpuMat res;
    if (float(dst_img.rows) >= float(dst_img.cols) * 3) { // 1.5
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());        
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        res.upload(srcCopy);
        return res;
    } else {
        return dst_img;
    }
}


cv::cuda::GpuMat Utility::GetCropImage(const cv::cuda::GpuMat &srcImage, const std::vector<std::vector<int>>& box) {
    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));
    return srcImage(cv::Rect(left, top, right - left, bottom - top));
}

std::vector<int> Utility::argsort(const std::vector<float> &array) {
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(), [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

    return array_index;
}

} // namespace OCR