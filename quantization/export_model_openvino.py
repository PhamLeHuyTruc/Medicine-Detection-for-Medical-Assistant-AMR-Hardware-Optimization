from ultralytics import YOLO

try:
    model = YOLO("D:\Intel_Target\Medicine\model\weight.pt")
except Exception as e:
    print(f"Error load model Pytorch")
    exit()
path = model.export(format='openvino', dynamic=True) 

print(f"Success")