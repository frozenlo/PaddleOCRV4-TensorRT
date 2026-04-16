# PaddleOCRv4_TensorRT HTTP Service

This document explains how to use the HTTP service functionality added to the PaddleOCRv4_TensorRT project.

## Building the Service

To build the HTTP service, follow these steps:

1. Make sure you have all the dependencies installed as mentioned in the main README.md
2. Build the project with CMake:

```bash
mkdir build
cd build
cmake ..
make -j
```

This will create an `ocr_service` executable in addition to the existing `run_inference_benchmark`.

## Running the Service

To run the OCR service with default configuration:

```bash
./ocr_service
```

To run the OCR service with a specific configuration file:

```bash
./ocr_service --config ../config.json
```

Or with explicit model paths:

```bash
./ocr_service --det_trt_model ../models/det_model.trt --rec_trt_model ../models/rec_model.trt --label_file ../models/ppocr_keys_v1.txt --port 8080
```

## Configuration

The service can be configured using a `config.json` file. The default configuration file looks like this:

```json
{
  "model_dir": "../models",
  "det_trt_model": "{model_dir}/det_model.trt",
  "det_onnx_model": "{model_dir}/det_model.onnx",
  "rec_trt_model": "{model_dir}/rec_model.trt",
  "rec_onnx_model": "{model_dir}/rec_model.onnx",
  "label_file": "{model_dir}/ppocr_keys_v1.txt",
  "default_port": 8080,
  "log_level": "info",
  "use_onnx": false
}
```

Configuration values can be overridden with command-line arguments.

## Command line arguments:
- `--config` or `-c`: Path to the configuration file (default: config.json)
- `--model_dir` or `-md`: Directory containing all model files
- `--det_trt_model` or `-dtm`: Path to the detection TensorRT model
- `--det_onnx_model` or `-dom`: Path to the detection ONNX model
- `--rec_trt_model` or `-rtm`: Path to the recognition TensorRT model
- `--rec_onnx_model` or `-rom`: Path to the recognition ONNX model
- `--label_file` or `-lf`: Path to the label file
- `--port` or `-p`: Port to run the service on (default is 8080)

## Model Types

This project supports both TensorRT (.trt) and ONNX (.onnx) models. You can switch between them by setting the `use_onnx` configuration option:
- `use_onnx: false` (default) - Use TensorRT models for maximum performance
- `use_onnx: true` - Use ONNX models for easier deployment and debugging

When `use_onnx` is set to true, the service will automatically use the ONNX model paths instead of the TRT model paths.

## Using the Service

The service exposes two endpoints:

### Health Check
GET `/health`

Example:
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "service": "PaddleOCR TensorRT Service",
  "use_onnx": false
}
```

### OCR Processing
POST `/ocr`

This endpoint accepts a multipart form data request with an image file.

Example using curl:
```bash
curl -X POST http://localhost:8080/ocr \
  -F "image=@path/to/your/image.jpg" \
  -H "Content-Type: multipart/form-data"
```

Response:
```json
{
  "status": "success",
  "results": [
    {
      "texts": ["Recognized", "text", "from", "image"],
      "confidence": 0.95
    }
  ]
}
```

If there's an error, you'll receive a response like:
```json
{
  "error": "Description of the error"
}
```

## Dependencies

The HTTP service uses these additional libraries:
1. [cpp-httplib](https://github.com/yhirose/cpp-httplib) - A single header-only C++ HTTP/HTTPS server and client library
2. [nlohmann/json](https://github.com/nlohmann/json) - JSON for Modern C++

These are automatically downloaded and included when you build the project.

## Troubleshooting

### "Unable to build or load TensorRT engine" error

This error typically occurs when:
1. The model files don't exist at the specified paths
2. The model files are in ONNX format but you're trying to load them as TRT engines
3. The model files are corrupted or incompatible

To resolve this issue:

1. Make sure the model files exist:
   ```bash
   ls -la ../models/
   ```

2. If you only have ONNX models and want to use them directly:
   ```bash
   ./run_inference_benchmark --config ../config.json --use_onnx true
   ```

3. If you want to use TensorRT models, you can convert ONNX models to TRT format:
   ```bash
   ./run_inference_benchmark --config ../config.json
   ```
   
   This will generate TRT engine files that can be used with the service.

4. Once you have the appropriate model files, run the service:
   ```bash
   ./ocr_service --config ../config.json
   ```