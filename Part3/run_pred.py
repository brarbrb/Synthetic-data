from ultralytics import YOLO
import cv2, os
from tqdm import tqdm

# This script is used only

model = YOLO("Part2/runs/pose/v8-small-add-data/weights/best.pt")  
img_dir = "detections/images"
lbl_dir = "detections/labels"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

video_path = "/datashare/project/vids_test/4_2_24_A_1.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id, saved_id = 0, 0

with tqdm(total=frame_count, desc="Processing frames") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            # save image
            img_name = f"frame_{frame_id:07d}.jpg"
            cv2.imwrite(os.path.join(img_dir, img_name), frame)
            saved_id += 1

        frame_id += 1
        pbar.update(1)   # update progress bar

cap.release()
print(f"Saved {saved_id} frames with YOLO labels")