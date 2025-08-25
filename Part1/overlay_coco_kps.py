import json, argparse, os
import cv2
import numpy as np
from typing import Dict, List, Tuple

# ------------------------------ COCO I/O ------------------------------

def load_coco(coco_path: str):
    with open(coco_path, "r") as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    cats_by_id = {c["id"]: c for c in coco.get("categories", [])}
    return coco, images_by_id, anns_by_image, cats_by_id

# ------------------------------ Helpers ------------------------------

def _auto_zero_based_skeleton(cat: Dict, num_kps: int) -> List[Tuple[int,int]]:
    """
    Normalize cat['skeleton'] to 0-based pairs.
    Heuristics:
      - If any index is 0 -> already 0-based.
      - Else if all indices in [1..num_kps] -> treat as 1-based and subtract 1.
      - Else: fall back to subtracting 1 if it keeps indices valid.
    """
    raw = cat.get("skeleton", [])
    if not raw:
        return []

    # Flatten to check range
    flat = [idx for pair in raw for idx in pair]
    if any(idx == 0 for idx in flat):
        # seems 0-based already
        sk0 = [(a, b) for a, b in raw if a >= 0 and b >= 0 and a < num_kps and b < num_kps]
        return sk0

    # likely 1-based if within [1..num_kps]
    if all(1 <= idx <= num_kps for idx in flat):
        sk0 = [(a-1, b-1) for a, b in raw if 1 <= a <= num_kps and 1 <= b <= num_kps]
        return sk0

    # fallback: try 1-based shift if it keeps valid
    sk_shift = [(a-1, b-1) for a, b in raw]
    if all(0 <= a < num_kps and 0 <= b < num_kps for a, b in sk_shift):
        return sk_shift

    # final fallback: keep as-is but clamp (best effort)
    sk0 = [(max(0, min(num_kps-1, a)), max(0, min(num_kps-1, b))) for a, b in raw]
    return sk0

def _cat_color(cat_id: int) -> Tuple[int,int,int]:
    """
    Stable BGR color per category id.
    """
    rng = np.random.RandomState(cat_id * 9973 + 12345)
    return tuple(int(x) for x in rng.randint(40, 220, size=3)[::-1])  # BGR

def _kp_color(vis_flag: int) -> Tuple[int,int,int]:
    """
    BGR per visibility flag:
      v=2 (visible): green
      v=1 (labeled but not visible): yellow
      v=0 (unlabeled/off): gray (rarely drawn)
    """
    if vis_flag >= 2:
        return (60, 200, 60)
    if vis_flag == 1:
        return (0, 220, 255)
    return (160, 160, 160)

def _safe_int_pair(x: float, y: float) -> Tuple[int,int]:
    return int(round(x)), int(round(y))

# ------------------------------ Drawing ------------------------------

def draw_overlay(img_bgr: np.ndarray,
                 anns: List[Dict],
                 cats_by_id: Dict[int, Dict],
                 alpha: float = 0.5,
                 draw_bbox: bool = True,
                 draw_skeleton: bool = True,
                 r: int = 4,
                 show_labels: bool = False,
                 mask_mode: str = "labeled"):
    """
    Returns (overlayed_image_bgr, mask_uint8)
      - overlayed_image_bgr: blended visualization
      - mask_uint8: binary mask built from keypoints+skeleton according to mask_mode:
            'labeled' -> v>0
            'visible' -> v==2
            'all'     -> draw everything (including v=0 if in range)
    """
    assert mask_mode in {"labeled", "visible", "all"}
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        cat = cats_by_id.get(ann.get("category_id"), {})
        kps = ann.get("keypoints", [])
        num_kps = len(kps) // 3

        # Per-category colors and data
        cat_color = _cat_color(int(ann.get("category_id", 0)))
        kp_names = cat.get("keypoints", []) if isinstance(cat.get("keypoints", []), list) else []

        # Build (x, y, v) triplets and validity
        pts = []
        for i in range(num_kps):
            x, y, v = kps[3*i:3*i+3]
            pts.append((float(x), float(y), int(v)))

        # Skeleton: normalize to 0-based pairs
        sk_pairs0 = _auto_zero_based_skeleton(cat, num_kps) if draw_skeleton else []

        # Draw skeleton first (lines under points)
        for (a, b) in sk_pairs0:
            if a < 0 or b < 0 or a >= num_kps or b >= num_kps:
                continue
            ax, ay, av = pts[a]
            bx, by, bv = pts[b]
            # show line if both are "present" by chosen mask_mode
            if mask_mode == "visible":
                cond = (av >= 2 and bv >= 2)
            elif mask_mode == "labeled":
                cond = (av > 0 and bv > 0)
            else:  # 'all'
                cond = True
            if cond:
                A = _safe_int_pair(ax, ay)
                B = _safe_int_pair(bx, by)
                cv2.line(overlay, A, B, cat_color, 2, lineType=cv2.LINE_AA)
                cv2.line(mask,   A, B, 255, 2, lineType=cv2.LINE_AA)

        # Draw keypoints
        for i, (x, y, v) in enumerate(pts):
            draw_this = (mask_mode == "all") or (mask_mode == "labeled" and v > 0) or (mask_mode == "visible" and v >= 2)
            if not draw_this:
                continue
            xi, yi = _safe_int_pair(x, y)
            cv2.circle(overlay, (xi, yi), r, _kp_color(v), -1, lineType=cv2.LINE_AA)
            cv2.circle(mask,   (xi, yi), r, 255, -1, lineType=cv2.LINE_AA)

            if show_labels:
                label = kp_names[i] if i < len(kp_names) else f"{i}"
                cv2.putText(overlay, label, (xi + 6, yi - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20,20,20), 2, cv2.LINE_AA)
                cv2.putText(overlay, label, (xi + 6, yi - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # Draw bbox
        if draw_bbox and "bbox" in ann:
            x, y, w_, h_ = ann["bbox"]
            p1 = _safe_int_pair(x, y)
            p2 = _safe_int_pair(x + w_, y + h_)
            cv2.rectangle(overlay, p1, p2, cat_color, 2)

    # Blend overlay onto original
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0.0)
    return blended, mask

# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="tmp/val/coco_keypoints.json", help="Path to COCO JSON")
    ap.add_argument("--images_root", default="tmp/val", help="Root folder containing the image files")
    ap.add_argument("--image_file", default="images/000001.png",
                    help="Relative path to the PNG inside images_root (e.g., images/000000.png)")
    ap.add_argument("--out_overlay", default="overlay.png", help="Where to save the overlay image")
    ap.add_argument("--out_mask", default="mask.png", help="Where to save the binary mask")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    ap.add_argument("--no_bbox", action="store_true", help="Do not draw bounding boxes")
    ap.add_argument("--no_skeleton", action="store_true", help="Do not draw skeletons")
    ap.add_argument("--mask_mode", choices=["labeled", "visible", "all"], default="all",
                    help="What to include in mask: labeled(v>0) / visible(v==2) / all")
    ap.add_argument("--show_labels", action="store_true", help="Draw keypoint names/indices")
    ap.add_argument("--radius", type=int, default=4, help="Keypoint circle radius")
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
        r=args.radius,
        show_labels=args.show_labels,
        mask_mode=args.mask_mode
    )

    cv2.imwrite(args.out_overlay, blended)
    cv2.imwrite(args.out_mask, mask)

    print(f"Saved overlay to: {args.out_overlay}")
    print(f"Saved keypoint/skeleton mask to: {args.out_mask}")
    print(f"(mask_mode={args.mask_mode}; bbox={'on' if not args.no_bbox else 'off'}; skeleton={'on' if not args.no_skeleton else 'off'})")

if __name__ == "__main__":
    main()
