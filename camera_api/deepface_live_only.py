import argparse
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

CAM_IP = "192.168.1.15"
RTSP_URL = f"rtsp://{CAM_IP}:554/live/ch00_0"
FRAME_W, FRAME_H = 640, 360
SCAN_EVERY = 30
FACE_SIMILARITY_THRESHOLD = 0.35
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live DeepFace face recognition and on-screen name display."
    )
    parser.add_argument(
        "--source",
        default=RTSP_URL,
        help="Camera source: RTSP string or camera index.",
    )
    parser.add_argument(
        "--db-path",
        default=str(Path(__file__).resolve().parent / "pictures"),
        help="Folder with person subfolders (e.g. pictures/pero, pictures/djuro).",
    )
    parser.add_argument(
        "--scan-every",
        type=int,
        default=SCAN_EVERY,
        help="How many frames to skip between DeepFace calls.",
    )
    return parser.parse_args()


def to_capture_source(raw_source: str):
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def cosine_similarity(vec1, vec2):
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-10
    if denom == 0:
        return -1.0
    return float(np.dot(vec1, vec2) / denom)


def load_reference_embeddings(db_path: Path):
    embeddings = []
    if not db_path.exists():
        print(f"[WARN] DB folder does not exist: {db_path}")
        return embeddings

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for image_path in db_path.rglob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue

        person_name = image_path.parent.name or image_path.stem
        try:
            representation = DeepFace.represent(
                img_path=str(image_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"[WARN] Cannot load reference {image_path}: {exc}")
            continue

        if not representation:
            continue

        if isinstance(representation, list):
            representation = representation[0]

        embedding = representation.get("embedding") if isinstance(representation, dict) else None
        if embedding is None:
            continue

        embeddings.append((person_name, np.asarray(embedding, dtype=np.float32)))

    print(f"[INFO] Loaded {len(embeddings)} reference embeddings")
    return embeddings


def recognize_face(frame, reference_embeddings):
    try:
        representation = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
    except Exception as exc:
        print(f"[WARN] DeepFace error: {exc}")
        return "Unknown", None

    if not representation:
        return "Unknown", None

    if isinstance(representation, list):
        representation = representation[0]

    embedding = representation.get("embedding") if isinstance(representation, dict) else None
    if embedding is None or not reference_embeddings:
        return "Unknown", None

    embedding = np.asarray(embedding, dtype=np.float32)
    best_person = "Unknown"
    best_score = -1.0

    for person_name, reference_embedding in reference_embeddings:
        score = cosine_similarity(embedding, reference_embedding)
        if score > best_score:
            best_score = score
            best_person = person_name

    if best_score < FACE_SIMILARITY_THRESHOLD:
        return "Unknown", None

    return best_person, best_score


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path).resolve()
    source = to_capture_source(args.source)

    print(f"[INFO] Source: {args.source}")
    print(f"[INFO] DB: {db_path}")
    print("[INFO] Loading reference embeddings...")
    reference_embeddings = load_reference_embeddings(db_path)

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open stream: {args.source}")

    cv2.namedWindow("DeepFace Live Only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DeepFace Live Only", FRAME_W, FRAME_H)

    frame_count = 0
    label = "Searching..."
    score_txt = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_count += 1
            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            if frame_count % max(1, int(args.scan_every)) == 0:
                small_frame = cv2.resize(frame, (320, 180))
                label, score = recognize_face(small_frame, reference_embeddings)
                score_txt = "" if score is None else f"score={score:.3f}"

            color = (0, 200, 0) if label != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"Face: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if score_txt:
                cv2.putText(frame, score_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("DeepFace Live Only", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
