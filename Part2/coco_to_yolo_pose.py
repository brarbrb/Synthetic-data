# converter of coco format to yolo format
import json, os, pathlib

def convert_split(split_dir, json_name="coco_keypoints.json", cat_id_to_cls=None):
    # cat_id_to_cls - to be more modular because coco_keypoints have another categories id 
    split_dir = pathlib.Path(split_dir)
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    lbl_dir.mkdir(exist_ok=True)

    coco = json.load(open(split_dir / json_name, "r"))
    imgs = {im["id"]: im for im in coco["images"]}
    # Map COCO category_id -> YOLO class index (0..N-1)
    if cat_id_to_cls is None:
        cats = sorted({a["category_id"] for a in coco["annotations"]})
        cat_id_to_cls = {cid:i for i,cid in enumerate(cats)}

    ann_by_img = {} # grouping annotations by image for mapping
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    for img_id, im in imgs.items():
        w, h = float(im["width"]), float(im["height"])
        label_lines = []
        for a in ann_by_img.get(img_id, []):
            cls = cat_id_to_cls[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            # bbox center + normalize
            cx = (x + bw/2.0) / w
            cy = (y + bh/2.0) / h
            nw = bw / w
            nh = bh / h

            kps = a.get("keypoints", [])
            # COCO kps are [x1,y1,v1,...] in ABS pixels; normalize
            kps_norm = []
            for i in range(0, len(kps), 3):
                kx = kps[i] / w
                ky = kps[i+1] / h
                kv = int(kps[i+2])
                
                # Clamp to [0,1] if numeric drift; keep visibility as is
                kx = min(max(kx, 0.0), 1.0)
                ky = min(max(ky, 0.0), 1.0)
                kps_norm.extend([kx, ky, kv])

            line = [cls, cx, cy, nw, nh] + kps_norm
            label_lines.append(" ".join(map(str, line)))

        # Write .txt with same stem as image
        img_rel = im["file_name"]  # e.g., "images/000123.png"
        txt_path = lbl_dir / (pathlib.Path(img_rel).stem + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(label_lines))

if __name__ == "__main__":
    # Train split
    convert_split("part3_data/train")
    # Val split
    convert_split("part3_data/val")
