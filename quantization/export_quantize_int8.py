from ultralytics import YOLO
import os

model_path = os.path.abspath('weight.pt')
data_yaml_path = os.path.abspath('data.yaml')
try:
    model = YOLO('weight.pt')
except Exception as e:
    print(f"Error load model Pytorch")
    exit() 

try:
    model.export(format='openvino', int8=True, data=data_yaml_path, imgsz=640)
    print("Success")
except Exception as e:
    print(f"\n--- {e} ---")
