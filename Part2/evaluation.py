from ultralytics import YOLO
import cv2, json, numpy as np, time


# === paths ===
MODEL_PATH = "runs/pose/train/weights/best.pt"   # change if needed
VIDEO_IN   = "/datashare/project/vids_test/4_2_24_A_1_small.mp4"
VIDEO_OUT  = "annotated_4_2_24_A_1.mp4"

# === your category-specific keypoints and skeleton ===
# IMPORTANT: class indices here must match model.names order (0..N-1).
# We'll map by the string name from model.names to be safe.
SKELETONS = {
    "tweezers":       [(0,1), (1,2), (0,3), (3,4)],
    "needle_holder":  [(0,2), (1,2), (2,3), (2,4)]
}
KEYPOINT_NAMES = {
    "tweezers":      ["handle_end", "left_arm", "left_tip", "right_arm", "right_tip"],
    "needle_holder": ["left_tip", "right_tip", "joint", "left_handle", "right_handle"]
}

# drawing helpers
def draw_pose(img, kxy, kconf=None, skeleton=None, labels=None, kp_thresh=0.25):
    h, w = img.shape[:2]
    # keypoints
    for j, (x, y) in enumerate(kxy):
        vis_ok = True
        if kconf is not None and kconf[j] is not None:
            vis_ok = float(kconf[j]) >= kp_thresh
        if not vis_ok: 
            continue
        if 0 <= int(x) < w and 0 <= int(y) < h:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            if labels and j < len(labels):
                cv2.putText(img, labels[j], (int(x)+4, int(y)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # skeleton
    if skeleton:
        for a, b in skeleton:
            if 0 <= a < len(kxy) and 0 <= b < len(kxy):
                xa, ya = kxy[a]
                xb, yb = kxy[b]
                if (0 <= int(xa) < w and 0 <= int(ya) < h and
                    0 <= int(xb) < w and 0 <= int(yb) < h):
                    cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)

def main():
    model = YOLO(MODEL_PATH)
    names = model.names  # dict like {0: 'tweezers', 1: 'needle_holder'} — verify!
    print("Class map:", names)

    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened(), f"Cannot open {VIDEO_IN}"
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
    # Inference loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run prediction on this frame
        results = model.predict(frame, imgsz=max(w, h), conf=0.25, iou=0.5, device=0, verbose=False)
        r = results[0]
        if len(results)>0:
            print("the results are: ", results)

        # draw detections
        if r.keypoints is not None and r.boxes is not None:
            kpts_xy   = r.keypoints.xy  # (N, K, 2) absolute pixels
            kpts_conf = r.keypoints.conf if hasattr(r.keypoints, "conf") else None  # (N, K)
            classes   = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(classes)):
                cls_idx = classes[i]
                cls_name = names.get(cls_idx, str(cls_idx))

                kxy = kpts_xy[i].cpu().numpy()
                kcf = kpts_conf[i].cpu().numpy() if kpts_conf is not None else None
                skel = SKELETONS.get(cls_name, None)
                labels = KEYPOINT_NAMES.get(cls_name, None)

                # draw box (optional)
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
                cv2.putText(frame, f"{cls_name} {float(r.boxes.conf[i]):.2f}",
                            (x1, max(y1-5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

                # draw kps & skeleton
                draw_pose(frame, kxy, kconf=kcf, skeleton=skel, labels=labels, kp_thresh=0.25)

        out.write(frame)
        # Optional live preview:
        # cv2.imshow("pose", frame)
        # if (cv2.waitKey(1) & 0xFF) == 27: break  # ESC

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"Saved: {VIDEO_OUT}")

if __name__ == "__main__":
    main()
