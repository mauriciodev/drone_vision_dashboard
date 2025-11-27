import asyncio
import json
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoImageProcessor, DetrForObjectDetection

app = FastAPI()

# Load Hugging Face DETR model
print("Loading Hugging Face model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model.eval()
print("Model ready on", device)


async def run_inference(image_bgr: np.ndarray, threshold: float = 0.9):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for DETR
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)

    # Run model inference (in thread to not block event loop)
    outputs = await asyncio.to_thread(model, **inputs)

    # Convert outputs (logits + boxes) to usable data
    target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score < threshold:
            continue
        box = box.detach().cpu().numpy().tolist()  # [x_min, y_min, x_max, y_max]
        detections.append({
            "score": float(score),
            "label_id": int(label),
            "label": model.config.id2label[int(label)],
            "box": [float(x) for x in box],
        })
    return detections


@app.websocket("/ws/detect")
async def detect_ws(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    try:
        while True:
            message = await ws.receive()
            if message.get("type") != "websocket.receive":
                break

            frame_bytes = message.get("bytes")
            if not frame_bytes:
                await ws.send_text(json.dumps({"error": "no_image"}))
                continue

            arr = np.frombuffer(frame_bytes, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                await ws.send_text(json.dumps({"error": "invalid_image"}))
                continue

            try:
                detections = await run_inference(image, threshold=0.9)
                response = {
                    "detections": detections,
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
                await ws.send_text(json.dumps(response))
            except Exception as e:
                await ws.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
 
