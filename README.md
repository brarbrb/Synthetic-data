# Computer Vision for Surgical Tools — 2D Pose Estimation

<p align="center">
  <b>Phase 1:</b> Synthetic Data &nbsp;•&nbsp;
  <b>Phase 2:</b> Train & Evaluate &nbsp;•&nbsp;
  <b>Phase 3:</b> Unsupervised Refinement on Real Video
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-✔-orange.svg)]()
[![BlenderProc](https://img.shields.io/badge/BlenderProc-✔-brightgreen.svg)]()
[![YOLO](https://img.shields.io/badge/YOLOv8-pose-success.svg)]()

---

## Project Intro

This repository develops a **2D pose estimation system** for surgical instruments using synthetic data first, then adapts to real, unlabeled videos.

- **Goal:** Detect per-tool keypoints and poses in 2D images/videos.
- **Key challenges:** Occlusion, articulation, domain gap, and real-time constraints.
- **Approach:** Synthetic data ➜ train on synthetic ➜ refine on unlabeled real video (self-training / pseudo-labels).

---

## 📁 Repository Structure 
```text
.
├─ Part1/                    
│  ├─ rendering_{type of rendering}.py     - there are multiple files that create different kinds of rendering
|  ├─ obj_features.py              - Exploring provided .obj and .mtl files
|  ├─ overlay_coco_kps.py         - draws key points on images of tools
|  └─ paste_on_random_background.py
├─ Part2/                      
│  ├─ train.py
│  ├─ data_config.yaml                      # YOLO/MMPose datasets (symlinks or copies)
│  └─ runs/                      # logs, checkpoints, metrics (automatic yolo logs) 
├─ Part3/                   # Phase 3: unsupervised domain adaptation
│  ├─ refine.py                  # pseudo-label loop
│  └─ runs_refined/
├─ inference/
│  ├─ predict.py                 # image inference
│  └─ video.py                   # video inference (OpenCV)
├─ requirements.txt
├─ README.md
├─ camera.json       - intrinsics for rendering
└─ LICENSE
```
Provided resources (we had on the VM in /datashare/project):

 - 3D CAD models (.obj) of tools (with articulation)

 - Backgrounds: COCO 2017; Textures/HDRI: Polyhaven

 - Unlabeled videos: (4_2_24_A_1.mp4, etc.)


# Synthetic-data

Installation of BlenderProc: 

```conda create -n synth python=3.10

conda activate synth

pip install blenderproc

blenderproc quickstart
```
# Best models for this task
1.  HRNet - Backbone that keeps high-resolution feature maps all the way through the network.

    Excellent accuracy for fine spatial localization (important for small keypoints like tool tips).

    There's open pose which is good for few objects on same image 

2. Top-down models (Keypoint R-CNN, YOLO-pose, RTMPose, ViTPose)

    First detect each object with a bounding box.

    Then run a pose head on the cropped image to get its keypoints.

    Usually more accurate for small or thin objects (like tweezers).

3.  Top-down models (Keypoint R-CNN, YOLO-pose, RTMPose, ViTPose)

    First detect each object with a bounding box.

    Then run a pose head on the cropped image to get its keypoints.

    Usually more accurate for small or thin objects (like tweezers).

### EDA results:
tweezers have only one material to work cand change, whilst the needle holder has two ptoperties that can vary.


Note: change location, scale, and rotation!


### Saving the dataset - correct format
For 2D pose estimation with two object classes (needle holder and tweezers) and synthetic data you’re generating yourself, the trick is to save the dataset in one general, model-agnostic format so you can later convert it to whatever a specific model requires.
That way, you only do the annotation once and can reformat as needed.

Our final structure is: 
1. For annotations: COCO format (JSON) – supports multiple objects per image, multiple keypoints per object, segmentation masks if needed.

2. We used next folder structure:
```
dataset/
│
├── images/
│   ├── train/
│   │   ├── img_00001.jpg
│   │   ├── img_00002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
│
├── annotations/
│   ├── train_coco.json   # COCO-style annotations for training
│   ├── val_coco.json
│   └── test_coco.json
│
└── meta/
    ├── class_names.txt   # ["needle_holder", "tweezers"]
    └── keypoints.txt     # ["tip", "joint", "handle_end", ...]
```

4. Our annotations were saved in coco-jsonlike this: 
```
{
  "images": [
    {"id": 1, "file_name": "img_00001.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "area": 1234,
      "iscrowd": 0,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...], 
      "num_keypoints": 5
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "needle_holder",
      "keypoints": ["tip_left", "tip_right", "joint", "handle_left", "handle_right"],
      "skeleton": [[1, 2], [2, 3], [3, 4], [3, 5]]
    },
    {
      "id": 2,
      "name": "tweezers",
      "keypoints": ["tip_left", "tip_right", "handle"],
      "skeleton": [[1, 3], [2, 3]]
    }
  ]
}
```

v in keypoints is visibility:

0 = not labeled,

1 = labeled but not visible,

2 = labeled & visible.

skeleton defines how the keypoints are connected in visualization.
