import cv2
import numpy as np
import time
from openvino import Core

# 1.Load configuration
MODEL_XML = "D:\Intel_Target\Medicine\model\medicine_weight_int8\medicine_weight.xml"  
CONF_THRESHOLD = 0.5                     
IOU_THRESHOLD = 0.4 

CLASS_NAMES = {
    0: "CHIEU",
    1: "SANG",
    2: "TOI"
}
CLASS_COLORS = {
    0: (255, 0, 0),   
    1: (0, 255, 0),   
    2: (0, 0, 255)    
}

def run_realtime_igpu():
    core = Core()
    model = core.read_model(MODEL_XML)
    compiled_model = core.compile_model(model, "GPU.0") 

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_h, original_w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (640, 640))
        image_blob = resized_frame / 255.0 
        input_tensor = np.expand_dims(image_blob.transpose(2, 0, 1), 0).astype(np.float32)

        # 2. INFERENCE
        start_time = time.perf_counter()
        results = compiled_model([input_tensor])[compiled_model.output(0)]
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

        # 3. PROCESSING
        outputs = results[0] 
        outputs = np.transpose(outputs) 

        boxes = []
        scores = []
        class_ids = []

        x_factor = original_w / 640.0
        y_factor = original_h / 640.0

        for row in outputs:
            classes_scores = row[4:]
            max_score = np.max(classes_scores)

            if max_score >= CONF_THRESHOLD:
                class_id = np.argmax(classes_scores)
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)


        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        current_counts = {0: 0, 1: 0, 2: 0}

        # 4. Insert result
        if len(indices) > 0:
            for i in indices.flatten() if isinstance(indices, np.ndarray) else indices:
                box = boxes[i]
                left, top, width, height = box
                score = scores[i]

                c_id = class_ids[i]
                current_counts[c_id] += 1

                label_name = CLASS_NAMES.get(c_id, f"Class_{c_id}")
                color = CLASS_COLORS.get(c_id, (255, 255, 255))

                cv2.rectangle(frame, (left, top), (left + width, top + height), color, 3)

                label_text = f"{label_name} {score:.2f}"
                cv2.putText(frame, label_text, (left, max(top - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        
        y_offset = 40
        for c_id, count in current_counts.items():
            label_name = CLASS_NAMES[c_id]
            color = CLASS_COLORS[c_id]
            
         
            display_count = min(count, 2)
            count_text = f"Tong {label_name}: {display_count} "
            cv2.putText(frame, count_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y_offset += 35


        cv2.putText(frame, f"iGPU Latency: {latency:.1f}ms | FPS: {fps:.1f}", (10, original_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Medical AMR - Realtime Detection (iGPU)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_igpu()