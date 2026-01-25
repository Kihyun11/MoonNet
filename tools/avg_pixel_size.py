import argparse, math
from pathlib import Path
from PIL import Image
import yaml

IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]

def parse_args():
    ap = argparse.ArgumentParser(description="Print average pixel size per class (and overall).")
    ap.add_argument("--root", required=True, help="Path to dataset root (folder with data.yaml)")
    ap.add_argument("--split", default="train", help="Split to analyze: train (default) or val")
    return ap.parse_args()

def find_image_for(stem: str, img_dir: Path):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def signed_area(xs, ys):
    s = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return 0.5 * s

def main():
    args = parse_args()
    root = Path(args.root)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    # Load class names (list or dict)
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list):
        raise ValueError("Could not parse class names from data.yaml")

    split = args.split
    lab_dir = root / split / "labels"
    img_dir = root / split / "images"
    if not lab_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Missing split directories: {img_dir} or {lab_dir}")

    per_class_areas = {i: [] for i in range(len(names))}
    all_areas = []
    files_processed = 0
    total_objs = 0

    for txt in sorted(lab_dir.glob("*.txt")):
        stem = txt.stem
        img_path = find_image_for(stem, img_dir)
        if img_path is None:
            continue
        with Image.open(img_path) as im:
            W, H = im.size

        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        files_processed += 1

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])

            # HBB: cls xc yc w h
            if len(parts) == 5:
                _, _, _, bw, bh = parts
                area = max(0.0, float(bw) * W * float(bh) * H)

            # OBB: cls x1 y1 x2 y2 x3 y3 x4 y4
            elif len(parts) == 9:
                coords = list(map(float, parts[1:]))
                xs = [coords[i] * W for i in range(0, 8, 2)]
                ys = [coords[i] * H for i in range(1, 8, 2)]
                area = abs(signed_area(xs, ys))

            else:
                continue

            per_class_areas[cid].append(area)
            all_areas.append(area)
            total_objs += 1

    # Pretty printer
    def fmt(area):
        a_int = int(round(area))
        side = int(round(math.sqrt(area)))
        return f"{a_int} px² (~{side} x {side})"

    print(f"\nAnalyzed split: {split}")
    print(f"Images with labels read: {files_processed}")
    print(f"Total objects counted:  {total_objs}\n")

    for cid, name in enumerate(names):
        vals = per_class_areas[cid]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{name:15s}: average pixel size = {fmt(avg)}, n={len(vals)}")
        else:
            print(f"{name:15s}: average pixel size = (no instances), n=0")

    if all_areas:
        overall = sum(all_areas) / len(all_areas)
        print(f"\nOverall:          average pixel size = {fmt(overall)}, n={len(all_areas)}")
    else:
        print("\nOverall:          average pixel size = (no instances)")

if __name__ == "__main__":
    main()