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
The project was splitted into 3 phases (parts):
1. Phase 1: dataset generation, domain gap analysis

2. Phase 2: model choice, training details, results

3. Phase 3: refinement strategy, comparison, challenges
---
## ⬇️ Weights & Artifacts

Phase 2 weights: <link>

Phase 3 weights: <link>

Videos:

results_synthetic_only.mp4

results_refined.mp4

## 📁 Repository Structure 
```text
.
├─ Part1/                    
│  ├─ rendering_{type of rendering}.py     - there are multiple files that create different kinds of rendering
|  ├─ obj_features.py              - Exploring provided .obj and .mtl files
|  ├─ overlay_coco_kps.py         - draws key points on images of tools
|  └─ paste_on_random_background.py
├─ Part2/                      
│  ├─ training_model.ipynb 
│  ├─ data_config.yaml
│  ├─ coco_to_yolo_pose.py        - converter from coco annotations to yolo
│  └─ runs/                      - logs, checkpoints, metrics (automatic yolo logs) 
├─ Part3/                   
│  ├─ refine.ipynb                  - pseudo-label loop
│  └─ runs_refined/
├─ inference/
│  ├─ predict.py                 - image inference
│  └─ video.py                   - video inference (OpenCV)
├─ requirements.txt
├─ README.md
├─ camera.json       - intrinsics for rendering
├─ sample images       - folder with examples of different renderings (.png files)
└─ LICENSE
```
Provided resources (we had on the VM in /datashare/project):

 - 3D CAD models (.obj) of tools (with articulation)

 - Backgrounds: COCO 2017; Textures/HDRI: Polyhaven

 - Unlabeled videos: (4_2_24_A_1.mp4, etc.)


### Saving the datasets
For 2D pose estimation with two object classes (needle holder and tweezers) and synthetic data it ws important to save the dataset in one general, model-agnostic format so we coudld later convert it to whatever a specific model requires.

Our final structure is: 
1. For annotations: COCO format (JSON) – supports multiple objects per image, multiple keypoints per object, segmentation masks if needed.

2. We used next folder structure: (we converted it later to yolo format) 
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
└── coco_keypoints.json COCO-style annotations for training
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
      "keypoints": ["joint","left_handle","left_tip","right_handle","right_tip"],
      "skeleton": [[0, 1], [0, 2], [0, 3], [0, 4]]   
    },
    {
      "id": 2,
      "name": "tweezers",
      "keypoints": ["handle_end","left_arm","left_tip","right_arm","right_tip"],
      "skeleton": [[0, 1], [0, 2], [1, 3], [2, 4]]
    }
  ]
}
```


# Synthetic-data

Installation of BlenderProc: 

```conda create -n synth python=3.10

conda activate synth

pip install blenderproc

blenderproc quickstart
```
