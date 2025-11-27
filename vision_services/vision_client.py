import cv2
import json
import time
from websocket import create_connection, ABNF

WS_URL = "ws://localhost:8000/ws/detect"

def main():
    ws = create_connection(WS_URL, timeout=10)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            ws.send(encoded.tobytes(), opcode=ABNF.OPCODE_BINARY)
            resp = json.loads(ws.recv())

            detections = resp.get("detections", [])
            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                label = det["label"]
                score = det["score"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.imshow("HF Object Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            time.sleep(0.03)
    finally:
        cap.release()
        ws.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 
