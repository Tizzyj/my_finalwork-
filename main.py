import argparse
import time
from collections import defaultdict

import cv2
import torch
from mediapipe_face import MediaPipeFace
from simple_tracker import IOUTracker
from model import FERModel
from fusion import TrackFusion
from utils import draw_overlay

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0, help="camera id or video file")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--model", default="", help="path to student checkpoint (optional)")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    mp_face = MediaPipeFace()
    tracker = IOUTracker(max_lost=15, iou_threshold=0.3)
    model = FERModel(device=device)
    if args.model:
        model.load_checkpoint(args.model)
    fusion = TrackFusion(window_size=9)

    font = cv2.FONT_HERSHEY_SIMPLEX
    warmup = 30
    frame_idx = 0
    timings = defaultdict(list)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        faces = mp_face.detect(frame)  # list of dict {box, landmarks, crop}
        detect_t = time.time()

        detections = []
        for f in faces:
            detections.append(f["box"])  # [x1,y1,x2,y2]
        track_results = tracker.update(detections)
        track_t = time.time()

        # map track id to face crop if IoU matched
        id_to_crop = {}
        for trk in track_results:
            tid = trk["id"]
            bbox = trk["box"]
            # find best face whose IoU > 0.3
            best = None
            for f in faces:
                x1,y1,x2,y2 = f["box"]
                # bbox equality approx by overlap center
                # simple center distance check
                bx = (x1+x2)/2; by=(y1+y2)/2
                if bx>bbox[0] and bx<bbox[2] and by>bbox[1] and by<bbox[3]:
                    best = f
                    break
            if best is not None:
                id_to_crop[tid] = best["crop"]

        infer_t0 = time.time()
        # inference per track (could batch)
        for tid, crop in id_to_crop.items():
            logits = model.predict_logits(crop)  # numpy logits
            fusion.push(tid, logits)
        infer_t1 = time.time()

        # get fusion results
        results = {}
        for trk in track_results:
            tid = trk["id"]
            if fusion.has_result(tid):
                label, conf = fusion.get_result(tid)
            else:
                label, conf = ("-", 0.0)
            results[tid] = {"box": trk["box"], "label": label, "conf": conf}

        draw_overlay(frame, results)

        end = time.time()
        frame_idx += 1
        if frame_idx > warmup:
            timings["detect"].append(detect_t-start)
            timings["track"].append(track_t-detect_t)
            timings["infer"].append(infer_t1-infer_t0)
            timings["total"].append(end-start)

        cv2.imshow("demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # print simple stats
    import numpy as np
    for k,v in timings.items():
        if v:
            print(f"{k}: mean {np.mean(v)*1000:.1f} ms, p95 {np.percentile(v,95)*1000:.1f} ms")

if __name__ == "__main__":
    main()