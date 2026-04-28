import cv2
import numpy as np
from ultralytics import YOLO
import torch
from camera import Camera
import threading
from pathlib import Path
import time
from retinaface import RetinaFace
from insightface.app import FaceAnalysis


face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(320, 320))

# ── Configuration ─────────────────────────────────────────
CAM_IP         = "192.168.50.222"
CAM_PORT       = 8899
CAM_USER       = "admin"
CAM_PASS       = "admin123"
RTSP_URL       = f"rtsp://{CAM_IP}:554/live/ch00_0"
MODEL_PATH     = "yolov8n-face.pt"
DB_PATH        = Path(__file__).resolve().parent / "pictures"

FRAME_W, FRAME_H = 960, 540
DEAD_ZONE      = 0.25
PAN_SPEED      = 0.5
TILT_SPEED     = 0.3
MOVE_DURATION  = 0.05
CONF_THRESHOLD = 0.5
RECOGNITION_EVERY = 15
ENABLE_FACE_RECOGNITION = False
FACE_SIMILARITY_THRESHOLD = 0.5
# ──────────────────────────────────────────────────────────

cam   = Camera(CAM_IP, CAM_PORT, CAM_USER, CAM_PASS)
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(RTSP_URL)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[INFO] YOLO inference device: {DEVICE_NAME}")

cv2.namedWindow("YOLOv8 Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Person Detection", FRAME_W, FRAME_H)

cam_busy = False
reference_embeddings = []


def cosine_similarity(vec1, vec2):
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    if denom == 0:
        return -1.0
    return float(np.dot(vec1, vec2) / denom)


def load_reference_embeddings(db_path: Path):
    embeddings = []

    for image_path in db_path.rglob("*"):
        if image_path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        faces = face_app.get(img)
        if len(faces) == 0:
            continue

        # uzmi NAJVEĆE lice (bitno!)
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

        emb = face.embedding
        name = image_path.parent.name

        embeddings.append((name, emb.astype(np.float32)))

    print(f"[INFO] Loaded {len(embeddings)} faces")
    return embeddings


def recognize_face(frame, reference_embeddings):
    faces = face_app.get(frame)

    if len(faces) == 0:
        return "Unknown", None

    face = max(faces, key=lambda f: f.det_score)
    emb = face.embedding.astype(np.float32)

    best_name = "Unknown"
    best_score = -1

    for name, ref in reference_embeddings:
        score = cosine_similarity(emb, ref)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < FACE_SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score

def crop_for_recognition(frame, box, padding=0.25):
    if box is None:
        return None

    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(w, x2 + pad_x)
    bottom = min(h, y2 + pad_y)

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (320, 320))


def move_camera(pan: float, tilt: float):
    global cam_busy
    if cam_busy:
        return
    def _move():
        global cam_busy
        cam_busy = True
        cam.move(pan=pan, tilt=tilt)
        time.sleep(MOVE_DURATION)
        cam.stop()
        cam_busy = False
    threading.Thread(target=_move, daemon=True).start()


def best_detection(results, frame_w, frame_h):
    """Return bounding box of the largest valid person detection."""
    best_box  = None
    best_area = 0
    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0 or conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            '''
            if bw > frame_w * 0.85 or bh > frame_h * 0.85:
                continue
            '''
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)
    return best_box


def put_text_right(img, text, row, color=(255, 255, 255), font_scale=1.1, thickness=2):
    """Draw text anchored to the right side of the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = img.shape[1] - tw - 20       
    y = 50 + row * 45                 
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


if not cap.isOpened():
    print("Error: could not open camera stream.")
else:
    print("[INFO] Running YOLOv8 Person Detection. Press 'q' to quit.")
    print(f"[INFO] Face DB: {DB_PATH}")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1

        h, w = frame.shape[:2]
        results = model(frame, conf=CONF_THRESHOLD, device=DEVICE)

        annotated = frame.copy()
        person_count = 0
        face_label = "Disabled" if not ENABLE_FACE_RECOGNITION else "Searching..."
        face_distance = None

        best_box = best_detection(results, w, h)

        # Draw all detected persons
        for result in results:
            for detection_box in result.boxes:
                cls  = int(detection_box.cls[0])
                conf = float(detection_box.conf[0])
                if cls != 0 or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, detection_box.xyxy[0])
                bw = x2 - x1
                bh = y2 - y1

                person_count += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"Person {person_count} {conf:.2f}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        if ENABLE_FACE_RECOGNITION and best_box is not None and frame_count % RECOGNITION_EVERY == 0:
            if not reference_embeddings:
                print("[INFO] Loading DeepFace reference embeddings...")
                reference_embeddings = load_reference_embeddings(DB_PATH)


                face_label, face_distance = recognize_face(frame, reference_embeddings)

        cx, cy, pan, tilt = 0.0, 0.0, 0.0, 0.0

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(annotated, "TRACKING",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cx = ((x1 + x2) / 2) / w - 0.5
            cy = ((y1 + y2) / 2) / h - 0.5

            if abs(cx) > DEAD_ZONE:
                pan = max(-1.0, min(1.0, cx * PAN_SPEED * 2.5))

            if abs(cy) > DEAD_ZONE:
                tilt = max(-1.0, min(1.0, -cy * TILT_SPEED * 2.5))

            if pan != 0.0 or tilt != 0.0:
                move_camera(pan, tilt)

            centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.drawMarker(annotated, centre, (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)

        # ── Info overlay – top right ───────────────────────
        put_text_right(annotated, f"Persons: {person_count}",  row=0, color=(255, 255, 255))
        put_text_right(annotated, f"Face: {face_label}", row=1, color=(0, 200, 0) if face_label != "Unknown" else (0, 0, 255))
        put_text_right(annotated, f"cx: {cx:.2f}  cy: {cy:.2f}", row=2, color=(0, 255, 255))
        put_text_right(annotated, f"pan: {pan:.2f}  tilt: {tilt:.2f}", row=3, color=(0, 255, 255))
        if face_distance is not None:
            put_text_right(annotated, f"dist: {face_distance:.4f}", row=4, color=(255, 255, 0))

        cv2.imshow("YOLOv8 Person Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection complete.")