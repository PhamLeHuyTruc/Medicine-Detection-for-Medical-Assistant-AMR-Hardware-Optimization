# Medicine-Detection-for-Medical-Assistant-AMR & Hardware-Optimization
This project focuses on the development and optimization of a computer vision system for a Medical Assistance Autonomous Mobile Robot (AMR). The primary goal is to achieve real-time medicine recognition and patient monitoring by leveraging hardware acceleration on Intel integrated GPUs (iGPU).

---
## Key Performance Highlights
- Architecture: YOLOv11 (Ultralytics) trained for medical object detection.

- Optimization Engine: Intel OpenVINO Toolkit.

- Quantization: FP32 to INT8 (Bit-depth reduction for massive speedup).

- Hardware: Optimized for Intel UHD Graphics 630 (on Dell Precision 7530).


##  Benchmark & Evaluation

| Model Strategy            | Device | mAP50 (Accuracy) | Latency (ms) | Throughput (FPS) |
| ------------------------- | ------ | ---------------- | ------------ | ---------------- |
| Baseline PyTorch (FP32)   | CPU    | 0.9905           | ~84.80       | ~11.79           |
| OpenVINO Optimized (INT8) | CPU    | 0.9898           | ~40.74       | ~24.54           |
| OpenVINO + iGPU (INT8)    | iGPU   | 0.9898           | ~22.51       | ~44.43           |

---
##  Optimization Workflow

To ensure the Medical Assistant AMR can recognize medications in real-time with minimal latency, I implemented a systematic optimization pipeline:

### 1. Model Training & Preparation
* **Architecture:** Trained a custom **YOLOv11** model specifically for medical items (medicine boxes, pill bottles, and medical tools).
* **Dataset:** Curated a high-quality dataset with medical-grade labels.
* **Export:** Converted the baseline PyTorch model (`.pt`) to the OpenVINO Intermediate Representation (IR) format.

### 2. INT8 Quantization 
To maximize throughput on edge hardware, I applied **INT8 Quantization** using the OpenVINO NNCF (Neural Network Compression Framework):
* **Goal:** Reduce model precision from FP32 to INT8 to decrease memory bandwidth and speed up computation.
* **Result:** Achieved a **~54% reduction** in latency with a negligible accuracy drop (only **0.48%**).

### 3. Hardware Acceleration (iGPU)
Most standard laptops/AMR controllers struggle with CPU-only inference. I optimized the system to offload the heavy AI workload to the **Intel® UHD Graphics 630 (iGPU)**

### 4. Performance Validation
I utilized a dual-metric validation approach:
* **Accuracy Check:** Running `model.val()` on the dedicated validation set to ensure medicine detection remains reliable.
* **Hardware Benchmark:** Using the Intel `benchmark_app` to measure the "Raw Power" of the iGPU, ensuring a stable **~44.43 FPS** across different lighting conditions.

---
## Tech Stack
- Language: Python 3.10

- Libraries: ultralytics, openvino, numpy

- Tools: Intel OpenVINO Benchmark App, TensorBoard

- Robotics: ROS 2 (SLAM & Nav2)
## Conclusion
Through systematic optimization, this project demonstrates that high-performance AI perception is achievable on standard workstation hardware without the need for discrete NVIDIA GPUs. This significantly reduces the power consumption and cost of the Medical AMR while maintaining "Real-Time" responsiveness.


