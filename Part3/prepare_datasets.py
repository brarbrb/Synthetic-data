import os
import shutil

# your good frames list
frames = [230, 231, 232, 234, 235, 236, 237, 294, 302, 510, 511, 512, 513, 843, 888, 891, 1508, 1536,
          1538, 1539, 1553, 1796,1801, 1802, 1804, 1813, 1916, 1919, 1920, 1921, 1922, 1923, 1924, 
          1931, 1932, 1936, 1938, 1939, 1969, 1971, 1972, 1973, 1974, 1975, 2066, 2097, 2070, 2093, 
          2537, 2553, 2568, 2582, 2583, 2585, 2586, 2588, 2594, 2603, 2605, 2613, 2613, 2645, 2653, 
          2654, 2655, 2666, 2672, 2678, 2681, 2683, 2684, 2685, 2687, 2693, 2703, 2712, 2713, 2714, 
          2716, 2617, 2720, 2722, 2742, 5172, 5078, 5073, 5074, 5053, 5012, 5013, 5014, 5015, 5016, 
          2017, 5018, 5019, 5020, 5021, 4847, 4856, 4859, 4819, 4247, 4196, 3948, 3949, 3805, 3806, 
          3807, 3808, 3809, 3810, 3811, 3812, 3813, 3788, 3789, 3791, 3792, 3793, 3500, 3501, 3503, 
          3505, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3497, 3498, 3499]

# paths
src_dir = "Part3/fine_tune/pseudo_v1/labels"
dst_dir = "Part3/detections/labels"

# make sure destination exists
os.makedirs(dst_dir, exist_ok=True)

for f in frames:
    # format frame number with 7 digits
    fname = f"frame_{f:07d}.txt"
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"⚠️ Label file not found: {src}")


img_dir = "Part3/detections/images"
lbl_dir = "Part3/detections/labels"

for img in os.listdir(img_dir):
    if img.endswith((".jpg", ".png")):
        base = os.path.splitext(img)[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        img_path = os.path.join(img_dir, img)
        if not os.path.exists(lbl_path):
            os.remove(img_path)
            print(f"Deleted image without label: {img_path}")
