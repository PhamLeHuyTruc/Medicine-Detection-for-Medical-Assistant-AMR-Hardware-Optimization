import time
import pandas as pd
import numpy as np
import openvino as ov
from ultralytics import YOLO

DATA_YAML = 'data.yaml'
MODEL_PT_PATH = 'medical_weight.pt'
MODEL_OV_PATH = 'medical_weight_int8_openvino_model'

def run_full_benchmark():
    results = []

    
    # Stage 1: ACCURACY 
    
    print("\n[1/3] Đang đo Accuracy bản gốc (.pt)...")
    model_pt = YOLO(MODEL_PT_PATH)
    res_pt = model_pt.val(data=DATA_YAML, verbose=False)
    mAP_pt = res_pt.results_dict['metrics/mAP50(B)']

    print("[2/3] Đang đo Accuracy bản OpenVINO INT8...")
    model_ov = YOLO(MODEL_OV_PATH, task='detect')
    res_ov = model_ov.val(data=DATA_YAML, verbose=False)
    mAP_ov = res_ov.results_dict['metrics/mAP50(B)']

    diff = (mAP_pt - mAP_ov) * 100
    print(f"--- ĐỘ LỆCH ĐỘ CHÍNH XÁC: {diff:.2f}% ---\n")

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: ĐO SPEED (Dùng Dummy Input để đo Raw Power)
    # ---------------------------------------------------------
    
    # 1. Test Original PT trên CPU
    print("--- Đang test Speed: Original PyTorch (CPU) ---")
    latencies_pt = []
    for _ in range(10): model_pt(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False) # Warm-up
    for _ in range(100):
        s = time.perf_counter()
        model_pt(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        latencies_pt.append((time.perf_counter() - s) * 1000)
    results.append(format_entry("Original PyTorch (FP32)", mAP_pt, latencies_pt))

    # 2. Test OpenVINO INT8 trên CPU
    print("--- Đang test Speed: OpenVINO INT8 (CPU) ---")
    latencies_ov_cpu = []
    for _ in range(10): model_ov(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False) # Warm-up
    for _ in range(100):
        s = time.perf_counter()
        model_ov(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        latencies_ov_cpu.append((time.perf_counter() - s) * 1000)
    results.append(format_entry("OpenVINO INT8 (CPU)", mAP_ov, latencies_ov_cpu))

    # 3. Test OpenVINO INT8 trên iGPU (Native OpenVINO)
    print("--- Đang test Speed: OpenVINO INT8 (iGPU) ---")
    core = ov.Core()
    ov_model = core.read_model(f"{MODEL_OV_PATH}/medical_weight.xml")
    compiled_model = core.compile_model(ov_model, "GPU.0")
    infer_req = compiled_model.create_infer_request()
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    
    latencies_ov_gpu = []
    for _ in range(10): infer_req.infer({0: dummy_input}) # Warm-up
    for _ in range(100):
        s = time.perf_counter()
        infer_req.infer({0: dummy_input})
        latencies_ov_gpu.append((time.perf_counter() - s) * 1000)
    results.append(format_entry("OpenVINO INT8 (iGPU)", mAP_ov, latencies_ov_gpu))

    # ---------------------------------------------------------
    # GIAI ĐOẠN 3: XUẤT BÁO CÁO
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    print("\n" + "="*85)
    print("BÁO CÁO TỔNG HỢP: ACCURACY VS PERFORMANCE")
    print("="*85)
    print(df.to_string(index=False))
    
    df.to_csv("benchmark_full_accuracy_speed.csv", index=False)
    print("\nĐã lưu báo cáo vào file benchmark_full_accuracy_speed.csv")

def format_entry(name, mAP, latencies):
    avg_lat = np.mean(latencies)
    return {
        "Model Strategy": name,
        "mAP50": f"{mAP:.4f}",
        "Avg Latency": f"{avg_lat:.2f} ms",
        "Throughput": f"{1000/avg_lat:.2f} FPS"
    }

if __name__ == "__main__":
    run_full_benchmark()