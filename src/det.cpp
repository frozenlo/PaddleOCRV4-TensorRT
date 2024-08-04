#include "det.h"

namespace OCR {

void TextDetect::Model_Init(std::string_view det_engine_path, std::string_view det_onnx_path) {
    bool succ = false;
    if (!det_engine_path.empty()) {
        if (Util::doesFileExist(det_engine_path.data())) {
            succ = this->loadNetwork(det_engine_path.data());
            if (!succ) {
                throw std::runtime_error("Unable to build or load TensorRT engine.");
            }
        }
    }

    if (!succ) {
        if (!det_onnx_path.empty() && Util::doesFileExist(det_onnx_path.data())) {            
            succ = buildLoadNetwork(det_onnx_path.data());
        }
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    }
}

void TextDetect::Model_Infer(const cv::cuda::GpuMat &gpuImg, vector<vector<vector<int>>> &boxes, vector<double> &times) {

    ////////////////////// preprocess ////////////////////////
    float ratio_h{}; // = resize_h / h
    float ratio_w{}; // = resize_w / w

    auto preprocess_start = std::chrono::steady_clock::now();
    cv::cuda::GpuMat resizedImg;

    // cpu resize result different from gpu resize, take care.
    /*cv::Mat tmp;
    cv::Mat img;
    gpuImg.download(img);
    ResizeImgType0_Cpu resize_op_cpu;
    resize_op_cpu.Run(img, tmp, this->max_side_len_, ratio_h, ratio_w);
    resizedImg.upload(tmp);*/

    this->resize_op_.Run(gpuImg, resizedImg, this->max_side_len_, ratio_h, ratio_w);

    this->normalize_op_.Run(resizedImg, this->mean_, this->scale_, true);
    this->permute_op_.Run(resizedImg);

    //////////////////////// inference //////////////////////////
    // void *buffers[2];
    //  为buffer[0]指针（输入）定义空间大小
    // float *inBlob = new float[1 * 3 * resize_img.rows * resize_img.cols];
    // this->permute_op_.Run(&resize_img, inBlob);

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option

    size_t batchSize = m_options.optBatchSize;
    auto preprocess_end = std::chrono::steady_clock::now();

    auto inference_start = std::chrono::steady_clock::now();
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    const auto numInputs = m_inputDims.size();
    // Preprocess all the inputs

    nvinfer1::Dims4 inputDims2 = {(int64_t)batchSize, 3, resizedImg.rows, resizedImg.cols};
    m_context->setInputShape(m_IOTensorNames[0].c_str(), inputDims2);

    m_buffers[0] = resizedImg.ptr<void>(0);

    int outputLength = resizedImg.cols * resizedImg.rows;
    Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[1], outputLength * m_options.maxBatchSize * sizeof(float), inferenceCudaStream));

    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Set the address of the input and output buffers
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
        if (!status) {
            return;
        }
    }

    // m_context->setBindingDimensions(inputIndex, in_dims); // 根据输入图像大小更新输入维
    m_context->enqueueV3(inferenceCudaStream);

    cv::cuda::GpuMat result(resizedImg.rows, resizedImg.cols, CV_32FC1, m_buffers[1]);

    cv::Mat pred;
    result.download(pred);

    cv::cuda::GpuMat mask;

    cv::cuda::threshold(result, mask, det_db_thresh_, 1, cv::THRESH_BINARY);

    mask.convertTo(mask, CV_8UC1, 255.0f);

    cv::Mat bitmap;
    mask.download(bitmap);

    //cv::imwrite("test1.png", bitmap);

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

    auto inference_end = std::chrono::steady_clock::now();

    ///////////////////// postprocess //////////////////////
    auto postprocess_start = std::chrono::steady_clock::now();

    boxes = post_processor_.BoxesFromBitmap(pred, bitmap, this->det_db_box_thresh_, this->det_db_unclip_ratio_, this->use_polygon_score_);

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, gpuImg); // 将resize_img中得到的bbox 映射回srcing中的bbox

    /*cv::Mat showImg = img.clone();
    for (auto &b : boxes) {

        cv::Point2i lt = cv::Point2i(b[0][0], b[0][1]);
        cv::Point2i rt = cv::Point2i(b[1][0], b[1][1]);
        cv::Point2i rb = cv::Point2i(b[2][0], b[2][1]);
        cv::Point2i lb = cv::Point2i(b[3][0], b[3][1]);

        cv::line(showImg, lt, rt, cv::Scalar(0, 0, 255));
        cv::line(showImg, rt, rb, cv::Scalar(0, 0, 255));
        cv::line(showImg, rb, lb, cv::Scalar(0, 0, 255));
        cv::line(showImg, lb, lt, cv::Scalar(0, 0, 255));
    }

    cv::imwrite("test.png", showImg);*/
    auto postprocess_end = std::chrono::steady_clock::now();
    std::cout << "Detected boxes num: " << boxes.size() << endl;

    std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
    times.push_back(double(preprocess_diff.count() * 1000));
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    times.push_back(double(inference_diff.count() * 1000));
    std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
    times.push_back(double(postprocess_diff.count() * 1000));

}

TextDetect::~TextDetect() {}

} // namespace OCR
