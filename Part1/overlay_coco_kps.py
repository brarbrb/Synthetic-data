import json, argparse, os
import cv2
import numpy as np

def load_coco(coco_path):
    with open(coco_path, "r") as f:
        coco = json.load(f)
    # Index by id for quick lookup
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    cats_by_id = {c["id"]: c for c in coco.get("categories", [])}
    return coco, images_by_id, anns_by_image, cats_by_id

def draw_overlay(img_bgr, anns, cats_by_id, alpha=0.5, draw_bbox=True, draw_skeleton=True, r=4):
    """Returns an overlayed BGR image and a binary mask (uint8: 0/255) of kps+skeleton."""
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        cat = cats_by_id.get(ann.get("category_id"), {})
        kps = ann.get("keypoints", [])
        # COCO keypoints are in (x, y, v) triplets
        pts = []
        for i in range(0, len(kps), 3):
            x, y, v = kps[i], kps[i+1], kps[i+2]
            if v > 0:  # v=0 not labeled, 1 labeled but not visible, 2 labeled & visible (we'll draw both 1 and 2)
                pts.append((int(round(x)), int(round(y)), int(v)))
        # Draw keypoints
        for (x, y, v) in pts:
            cv2.circle(overlay, (x, y), r, (0, 255, 255), -1)       # on overlay
            cv2.circle(mask,   (x, y), r, 255, -1)                  # on mask (white)

        # Draw skeleton if provided
        if draw_skeleton and "skeleton" in cat:
            # COCO keypoint indices in skeleton are 1-based
            for a, b in cat["skeleton"]:
                a_idx, b_idx = a - 1, b - 1
                if 0 <= 3*a_idx+2 < len(kps) and 0 <= 3*b_idx+2 < len(kps):
                    ax, ay, av = kps[3*a_idx:3*a_idx+3]
                    bx, by, bv = kps[3*b_idx:3*b_idx+3]
                    if av > 0 and bv > 0:
                        A = (int(round(ax)), int(round(ay)))
                        B = (int(round(bx)), int(round(by)))
                        cv2.line(overlay, A, B, (0, 165, 255), 2)
                        cv2.line(mask,   A, B, 255, 2)

        # Draw bbox if present
        if draw_bbox and "bbox" in ann:
            x, y, w_, h_ = ann["bbox"]
            p1 = (int(round(x)), int(round(y)))
            p2 = (int(round(x + w_)), int(round(y + h_)))
            cv2.rectangle(overlay, p1, p2, (255, 0, 255), 2)

    # Blend overlay onto original
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0.0)
    return blended, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="out/train/coco_keypoints.json", help="Path to coco.json")
    ap.add_argument("--images_root", default="out/train", help="Root folder containing the image files")
    ap.add_argument("--image_file", default= "images/000469.png", help="Relative path to the PNG inside images_root (e.g., images/000000.png)")
    ap.add_argument("--out_overlay", default="overlay.png", help="Where to save the overlay image")
    ap.add_argument("--out_mask", default="mask.png", help="Where to save the binary mask")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    ap.add_argument("--no_bbox", action="store_true", help="Do not draw bounding boxes")
    ap.add_argument("--no_skeleton", action="store_true", help="Do not draw skeletons")
    args = ap.parse_args()

    coco, images_by_id, anns_by_image, cats_by_id = load_coco(args.coco)

    # Find the image_id from the provided image_file (by matching file_name)
    target_rel = args.image_file.replace("\\", "/")
    img_id = None
    for _id, im in images_by_id.items():
        if im.get("file_name", "").replace("\\", "/") == target_rel:
            img_id = _id
            break
    if img_id is None:
        raise ValueError(f"Could not find image with file_name '{target_rel}' in {args.coco}")

    img_path = os.path.join(args.images_root, target_rel)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at '{img_path}'")

    anns = anns_by_image.get(img_id, [])
    blended, mask = draw_overlay(
        img_bgr,
        anns,
        cats_by_id,
        alpha=args.alpha,
        draw_bbox=(not args.no_bbox),
        draw_skeleton=(not args.no_skeleton),
        r=4
    )

    cv2.imwrite(args.out_overlay, blended)
    cv2.imwrite(args.out_mask, mask)

    print(f"Saved overlay to: {args.out_overlay}")
    print(f"Saved keypoint/skeleton mask to: {args.out_mask}")

if __name__ == "__main__":
    main()
