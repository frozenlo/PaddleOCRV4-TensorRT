# PaddleOCRv4_TensorRT
## Enviroment
* cuda 12.4
* cudnn 8
* tensorrt-cuda12.1-trt10.1.0.27
* opencv 4.10.0

## How to run
* modify the OpenCV_DIR, fmt_DIR, TensorRT_DIR, spglog_INCLUDE_DIRS path in **CmakeLists.txt**
* modify the path parameters.txt in **main.cpp**
```
 mkdir build
 cd build
 cmake ..
 make -j
 ./rec
```
## Result
![image](https://user-images.githubusercontent.com/87298337/179940172-4182773c-5786-4d5e-a1e1-8a63e98b4f10.png)

# Thanks to

- [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) for creating a easy-to-use TensorRT C++ API Tutorial.

- [PaddleOCRv2_TensorRT](https://github.com/zwenyuan1/PaddleOCRv2_TensorRT) for creating some C++ implemention of PaddleOCR preprocess and postprocess method.

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for creating awesome and practical OCR tools that help users train better models and apply them into practice.