Inference settings that often “wake up” detections
yolo pose predict model=best.pt source=your_video.mp4 \
  imgsz=1280 conf=0.15 iou=0.6 vid_stride=1 agnostic_nms=False save=True
