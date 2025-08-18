from ultralytics import YOLO
import cv2, json, numpy as np, time

model = YOLO("path/to/your_model.pt")   # your trained pose model
video_path = "4_2_24_A_1.mp4"
out_path   = "annotated_4_2_24_A_1.mp4"

cap = cv2.VideoCapture(video_path)
w, h = int(cap.get(3)), int(cap.get(4))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

per_frame_stats = []
t0 = time.time()
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t1 = time.time()
    res = model.predict(frame, imgsz=960, conf=0.25, verbose=False)[0]
    t2 = time.time()

    # Draw and write video
    plotted = res.plot()   # draws boxes + skeletons
    out.write(plotted)

    # Collect stats
    # keypoints shape: (num_instances, num_kpts, 3) -> (x,y,conf)
    kpts = res.keypoints.xy  if res.keypoints is not None else []
    kconfs = res.keypoints.conf if res.keypoints is not None else []
    num_people = len(kpts) if kpts is not None else 0
    inf_ms = (t2 - t1) * 1000
    per_frame_stats.append({
        "frame": frame_idx,
        "num_people": num_people,
        "inf_ms": inf_ms,
        "avg_kpt_conf": float(np.mean(kconfs)) if num_people else None
    })
    frame_idx += 1

cap.release(); out.release()
total_time = time.time() - t0

# Save stats
with open("pose_stats_4_2_24_A_1.json", "w") as f:
    json.dump({
        "video": video_path,
        "frames": frame_idx,
        "runtime_sec": total_time,
        "avg_fps": frame_idx / total_time,
        "per_frame": per_frame_stats
    }, f, indent=2)

print("Saved:", out_path, "and pose_stats_4_2_24_A_1.json")
