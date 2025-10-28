#pragma once
#include "rec.h"
#include "logger.h"


namespace OCR {

    void TextRec::Model_Init(std::string_view det_engine_path, std::string_view det_onnx_path) {
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

    uint32_t TextRec::getMaxOutputLength(nvinfer1::Dims tensorShape) const {
        return (m_options.MAX_DIMS_[3] + 4) / 8 * tensorShape.d[tensorShape.nbDims - 1] * rec_batch_num_;
    }

    float TextRec::Model_Infer(const cv::Mat& img, std::string& text){
        cv::cuda::GpuMat gpuImg;
        gpuImg.upload(img);
        return Model_Infer(gpuImg, text);
    }

    float TextRec::Model_Infer(const cv::cuda::GpuMat& img, std::string& text){
        cv::cuda::GpuMat resize_img;
        float max_wh_ratio = img.cols / (float)img.rows;
        this->resize_op_.Run(img, resize_img, max_wh_ratio);
        this->normalize_op_.Run(resize_img, this->mean_, this->scale_, true);
        this->permute_op_.Run(resize_img);
        int batch_width = int(m_options.OPT_DIMS_[2] * max_wh_ratio); // 这个batch里图像的宽度
        nvinfer1::Dims4 inputDims2 = { 1, 3, m_options.OPT_DIMS_[2], batch_width };
        m_context->setInputShape(m_IOTensorNames[0].c_str(), inputDims2);

        // Create stream
        cudaStream_t inferenceCudaStream;
        CHECK(cudaStreamCreate(&inferenceCudaStream));
        // m_context->nb
        auto tensorShape = m_outputDims[0];
        size_t outputWidth = (batch_width + 4) / 8;
        size_t outputLength = outputWidth * tensorShape.d[2];

        m_buffers[0] = resize_img.ptr<void>();
        for (size_t i = 0; i < m_buffers.size(); ++i) {
            bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
            if (!status) {
                return 0.f;
            }
        }
        m_context->enqueueV3(inferenceCudaStream);

        std::vector<float> imgRes;
        imgRes.resize(outputLength);

        Util::checkCudaErrorCode(cudaMemcpyAsync(imgRes.data(), static_cast<char*>(m_buffers[1]),
            outputLength * sizeof(float),
            cudaMemcpyDeviceToHost, inferenceCudaStream));
        cudaStreamSynchronize(inferenceCudaStream);
        //Util::checkCudaErrorCode(cudaFreeAsync(m_buffers[1], inferenceCudaStream));
        // Release stream and buffers
        cudaStreamDestroy(inferenceCudaStream);

        std::vector<std::string> str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;


        for (int n = 0; n < outputWidth; n++) { // n = 2*l + 1
            argmax_idx = int(Utility::argmax(imgRes.cbegin() + n * tensorShape.d[2],
                imgRes.cbegin() + (n + 1) * tensorShape.d[2]));
            max_value = float(*std::max_element(imgRes.cbegin() + n * tensorShape.d[2],
                imgRes.cbegin() + (n + 1) * tensorShape.d[2]));

            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                str_res.push_back(this->label_list_[argmax_idx]);
            }
            last_index = argmax_idx;
        }

        text = fmt::format("{}", fmt::join(str_res, ""));
        score /= count;
        return score;

    }

    void TextRec::Model_Infer(std::vector<cv::cuda::GpuMat>& img_list, std::vector<std::pair<std::vector<std::string>, double>>& rec_res, std::vector<int>& idx_map,
        std::vector<double>* times) {
        std::chrono::duration<float> preprocess_diff = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
        std::chrono::duration<float> inference_diff = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();
        std::chrono::duration<float> postprocess_diff = std::chrono::steady_clock::now() - std::chrono::steady_clock::now();

        int img_num = img_list.size();
        std::vector<float> width_list; // 存储所有待识别图像的宽高比
        for (int i = 0; i < img_num; i++)
            width_list.push_back(float(img_list[i].cols) / img_list[i].rows);

        std::vector<int> indices = Utility::argsort(width_list); // 对宽高比由小到大进行排序，并获取indices
        std::vector<int> copy_indices = indices;
        // 记录一个batch里识别结果为空的idx
        std::vector<int> nan_idx;

        for (int begin_img = 0; begin_img < img_num; begin_img += this->rec_batch_num_) {

            /////////////////////////// preprocess ///////////////////////////////
            auto preprocess_start = std::chrono::steady_clock::now();
            int end_img = std::min(img_num, begin_img + this->rec_batch_num_);
            //float max_wh_ratio = m_options.OPT_DIMS_[3] / (float)m_options.OPT_DIMS_[2];
            float max_wh_ratio = 0;
            for (int ino = begin_img; ino < end_img; ino++) {
                int h = img_list[indices[ino]].rows;
                int w = img_list[indices[ino]].cols;
                float wh_ratio = w * 1.0 / h;
                max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
            } // 找最大的宽高比

            std::vector<cv::cuda::GpuMat> norm_img_batch;
            for (int ino = begin_img; ino < end_img; ino++) {
                cv::cuda::GpuMat resize_img;
                this->resize_op_.Run(img_list[indices[ino]], resize_img, max_wh_ratio);
                this->normalize_op_.Run(resize_img, this->mean_, this->scale_, true);
                norm_img_batch.push_back(resize_img);
            } // 将一个batch里的img按照最大宽高比resize到高为48，宽为48*max_wh_ratio，并分别做归一化。


            ////////////////////////// 

            // 为buffer[0]指针（输入）定义空间大小
            // int batch_width = 1999;
            int batch_width = int(m_options.OPT_DIMS_[2] * max_wh_ratio); // 这个batch里图像的宽度
            cv::cuda::GpuMat dest;
            this->permute_batch_op_.Run(norm_img_batch, dest);

            auto preprocess_end = std::chrono::steady_clock::now();
            preprocess_diff += preprocess_end - preprocess_start;

            auto inference_start = std::chrono::steady_clock::now();
            nvinfer1::Dims4 inputDims2 = { (int64_t)norm_img_batch.size(), 3, m_options.OPT_DIMS_[2], batch_width };
            m_context->setInputShape(m_IOTensorNames[0].c_str(), inputDims2);

            // Create stream
            cudaStream_t inferenceCudaStream;
            CHECK(cudaStreamCreate(&inferenceCudaStream));


            // m_context->nb
            auto tensorShape = m_outputDims[0];
            size_t outputWidth = (batch_width + 4) / 8;
            size_t outputLength = outputWidth * tensorShape.d[2];

            //Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[1], rec_batch_num_ * outputLength * sizeof(float), inferenceCudaStream));
            m_buffers[0] = dest.ptr<void>();
            for (size_t i = 0; i < m_buffers.size(); ++i) {
                bool status = m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i]);
                if (!status) {
                    return;
                }
            }

            m_context->enqueueV3(inferenceCudaStream);

            std::vector < std::vector<float>> result;

            for (int img = 0; img < norm_img_batch.size(); ++img) {
                std::vector<float>& output = result.emplace_back(outputLength);
                output.resize(outputLength);

                Util::checkCudaErrorCode(cudaMemcpyAsync(output.data(), static_cast<char*>(m_buffers[1]) + (img * sizeof(float) * outputLength),
                    outputLength * sizeof(float),
                    cudaMemcpyDeviceToHost, inferenceCudaStream));
            }


            // 从gpu取数据到cpu上
            cudaStreamSynchronize(inferenceCudaStream);
            //Util::checkCudaErrorCode(cudaFreeAsync(m_buffers[1], inferenceCudaStream));
            // Release stream and buffers
            cudaStreamDestroy(inferenceCudaStream);
            auto inference_end = std::chrono::steady_clock::now();
            inference_diff += inference_end - inference_start;

            ////////////////////// postprocess ///////////////////////////
            auto postprocess_start = std::chrono::steady_clock::now();

            std::vector<int> predict_shape;
            predict_shape.push_back(norm_img_batch.size());
            predict_shape.push_back(outputWidth);
            predict_shape.push_back(tensorShape.d[2]);
            //for (int j = 0; j < out_dims.nbDims; j++)
            //    predict_shape.push_back(out_dims.d[j]);

            for (int m = 0; m < predict_shape[0]; m++) { // m = batch_size
                std::pair<std::vector<std::string>, double> temp_box_res;
                std::vector<std::string> str_res;
                int argmax_idx;
                int last_index = 0;
                float score = 0.f;
                int count = 0;
                float max_value = 0.0f;

                std::vector<float>& imgRes = result[m];

                for (int n = 0; n < predict_shape[1]; n++) { // n = 2*l + 1
                    argmax_idx = int(Utility::argmax(imgRes.cbegin() + n * predict_shape[2],
                        imgRes.cbegin() + (n + 1) * predict_shape[2]));
                    max_value = float(*std::max_element(imgRes.cbegin() + n * predict_shape[2],
                        imgRes.cbegin() + (n + 1) * predict_shape[2]));

                    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                        score += max_value;
                        count += 1;
                        str_res.push_back(this->label_list_[argmax_idx]);
                    }
                    last_index = argmax_idx;
                }
                score /= count;
                if (isnan(score)) {
                    nan_idx.push_back(begin_img + m);
                    continue;
                }

                temp_box_res.first = str_res;
                temp_box_res.second = score;
                rec_res.push_back(temp_box_res);
            }

            //delete[] outBlob;
            auto postprocess_end = std::chrono::steady_clock::now();
            postprocess_diff += postprocess_end - postprocess_start;
        }

        for (int i = nan_idx.size() - 1; i >= 0; i--) {
            copy_indices.erase(copy_indices.begin() + nan_idx[i]);
        }

        if (copy_indices.size() == rec_res.size()) {
            // cout<<"rec res size is equal to indices size"<<endl;
            idx_map = copy_indices;
        }
        if (times != NULL) {
            if (times->empty()) {
                times->push_back(double(preprocess_diff.count()));
                times->push_back(double(inference_diff.count()));
                times->push_back(double(postprocess_diff.count()));
            }
            else {
                times->at(0) += double(preprocess_diff.count());
                times->at(1) += double(inference_diff.count());
                times->at(2) += double(postprocess_diff.count());
            }
        }

    }
}