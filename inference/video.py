import argparse, os, sys, time
import cv2
from ultralytics import YOLO

def parse_source(s: str):
    return int(s) if s.isdigit() else os.path.abspath(os.path.expanduser(s))

def make_writer(path, width, height, fps, preferred=("mp4v", "avc1"), fallback=("XVID", "MJPG")):
    # Try MP4 first
    base, ext = os.path.splitext(path)
    for fourcc_str in preferred:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out_path = base + ".mp4"
        w = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if w.isOpened():
            print(f"[INFO] Using codec {fourcc_str} → {out_path}")
            return w, out_path
        else:
            print(f"[WARN] Failed to init MP4 with {fourcc_str}")

    # Fallback to AVI
    for fourcc_str in fallback:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out_path = base + ".avi"
        w = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if w.isOpened():
            print(f"[INFO] Using codec {fourcc_str} → {out_path}")
            return w, out_path
        else:
            print(f"[WARN] Failed to init AVI with {fourcc_str}")

    return None, None

def main():
    ap = argparse.ArgumentParser(description="YOLO (pose) video inference with robust writer.")
    ap.add_argument("--weights", type=str, default="best.pt", help="Path to weights")
    ap.add_argument("--source", type=str, required=True, help="Video path, URL, or webcam index (e.g., 0)")
    ap.add_argument("--save_dir", type=str, default="runs/video", help="Output directory")
    ap.add_argument("--out", type=str, default=None, help="Output base name (extension decided by codec)")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", type=str, default=None, help='cuda, cuda:0, or cpu')
    ap.add_argument("--show", action="store_true", help="Show preview window")
    args = ap.parse_args()

    weights = os.path.abspath(os.path.expanduser(args.weights))
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    source = parse_source(args.source)

    # Load model
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {weights}\n{e}")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {args.source}")
        sys.exit(1)

    # Probe fps safely
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # NaN check
        fps = 25.0
        print(f"[WARN] FPS unknown/zero; using default {fps}")

    # Build base name for output
    if isinstance(source, int):
        base_name = f"webcam_{source}"
    else:
        base_name = os.path.splitext(os.path.basename(str(source)))[0]
    if args.out:
        base_name = os.path.splitext(args.out)[0]

    out_base = os.path.join(save_dir, base_name)

    writer = None
    out_path = None
    frames = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False
            )
            plotted = results[0].plot()

            if writer is None:
                h, w = plotted.shape[:2]
                writer, out_path = make_writer(out_base, w, h, fps)
                if writer is None:
                    print("[ERROR] Failed to create any VideoWriter. Install codecs or try --device cpu.")
                    break
                print(f"[INFO] Writing to: {out_path}")

            writer.write(plotted)
            frames += 1

            if args.show:
                cv2.imshow("Predictions (press q to quit)", plotted)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    if frames == 0:
        print("[ERROR] No frames were written. Possible causes:\n"
              "  • Source path invalid or unreadable\n"
              "  • FPS reported as 0 and writer rejected\n"
              "  • Codec not available (try .avi fallback)\n"
              "  • Permission issue in save_dir")
    else:
        print(f"[INFO] Done. {frames} frames in {dt:.2f}s ({frames/max(dt,1e-6):.2f} FPS).")
        print(f"[INFO] Saved: {out_path}")
        if out_path and not os.path.exists(out_path):
            print("[WARN] Output path missing after write — check permissions/antivirus.")
