import argparse
import cv2
from ultralytics import YOLO
from tqdm import tqdm 


def run_video(weights, source, output):
    model = YOLO(weights)

    # Open video source (file or webcam index)
    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)
            pbar.update(1)
            
    cap.release()
    out.release()
    print(f"Saved predictions to {output}")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO predictions on a video and save output.")
    parser.add_argument("--weights", type=str, default="part3_best.pt", help="Path to YOLO weights file")
    parser.add_argument("--source", type=str, default="/datashare/project/vids_test/4_2_24_A_1.mp4", help="Path to input video or webcam index")
    parser.add_argument("--output", type=str, default="results_refined.mp4", help="Path to save output video")
    args = parser.parse_args()

    run_video(args.weights, args.source, args.output)