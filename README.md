# Synthetic-data

Installation of BlenderProc: 

```conda create -n synth python=3.10

conda activate synth

pip install blenderproc

blenderproc quickstart
```


### EDA results:
tweezers have only one material to work cand change, whilst the needle holder has two ptoperties that can vary.


Note: change location, scale, and rotation!


### Saving the dataset - correct format
For 2D pose estimation with two object classes (needle holder and tweezers) and synthetic data you’re generating yourself, the trick is to save the dataset in one general, model-agnostic format so you can later convert it to whatever a specific model requires.
That way, you only do the annotation once and can reformat as needed.

Our final structure is: 
1. For annotations: COCO format (JSON) – supports multiple objects per image, multiple keypoints per object, segmentation masks if needed.

2. We used next folder structure: 
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


3. Our annotations were saved in coco-jsonlike this: 
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
