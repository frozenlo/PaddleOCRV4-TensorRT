cmake_minimum_required(VERSION 3.18)
project(paddleocrv4-tensorrt)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


set(OpenCV_DIR "E:/do2024/opencv/out/install/x64-${CMAKE_BUILD_TYPE}")
set(fmt_DIR "E:/do2024/fmt-11.0.1/out/build/x64-${CMAKE_BUILD_TYPE}")
set(TensorRT_DIR "E:/do2024/TensorRT-10.1.0.27")
set(spglog_INCLUDE_DIRS "E:/do2024/spdlog-1.x/include")


# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR E:/do2024/TensorRT-10.1.0.27)
endif()
# Use the correct version of CUDA
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${spglog_INCLUDE_DIRS} include include/interfaces)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} ${spglog_LIBRARIES} fmt::fmt)

add_executable(run_inference_benchmark src/main.cpp src/det.cpp src/engine.cpp src/preprocess_op.cpp src/postprocess_op.cpp src/rec.cpp src/ocr.cpp src/utility.cpp  src/clipper.cpp)
target_include_directories(run_inference_benchmark PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${spglog_INCLUDE_DIRS} include include/interfaces)
#src/ocr.cpp src/rec.cpp src/utility.cpp src/clipper.cpp
target_link_libraries(run_inference_benchmark tensorrt_cpp_api fmt::fmt)