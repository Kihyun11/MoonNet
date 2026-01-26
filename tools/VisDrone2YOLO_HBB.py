# visdrone2yolo_hbb.py
import argparse, os, shutil
from pathlib import Path
from PIL import Image

VISDRONE_CLASSES = [
    "pedestrian","people","bicycle","car","van",
    "truck","tricycle","awning-tricycle","bus","motor"
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert VisDrone2019-DET to YOLO-HBB (axis-aligned) labels."
    )
    ap.add_argument("--root", required=True,
                    help="Path to VisDrone2019 root (contains VisDrone2019-DET-train/val/test-dev)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--splits", nargs="*", default=["train","val","test-dev"],
                    help="Splits to convert (subset of: train val test-dev)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy images into <out>/<split>/images (recommended on Windows)")
    ap.add_argument("--keep_empty", action="store_true",
                    help="Keep images with no valid objects")
    return ap.parse_args()

def link_or_copy(src: Path, dst: Path, force_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if force_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

def convert_split(split: str, root: Path, out: Path, copy_flag: bool, keep_empty: bool):
    src = root / f"VisDrone2019-DET-{split}"
    img_dir = src / "images"
    ann_dir = src / "annotations"  # test-dev may be empty
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing: {img_dir}")

    out_img = out / split / "images"
    out_lab = out / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    kept = 0
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        ann = ann_dir / f"{img.stem}.txt"

        # image size
        try:
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            continue

        lines = []
        if ann.exists():
            for raw in ann.read_text().splitlines():
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 8:
                    continue
                try:
                    x = float(parts[0]); y = float(parts[1])
                    w = float(parts[2]); h = float(parts[3])
                    cat = int(parts[5])
                except Exception:
                    continue

                # skip ignore regions (cat==0); map 1..10 -> 0..9
                if cat == 0:
                    continue
                if not (1 <= cat <= 10):
                    continue
                cls_id = cat - 1

                if w <= 0 or h <= 0:
                    continue

                xc = (x + w / 2.0) / W
                yc = (y + h / 2.0) / H
                bw = w / W
                bh = h / H
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines and not keep_empty:
            continue

        # place image & label
        link_or_copy(img, out_img / img.name, copy_flag)
        (out_lab / f"{img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        kept += 1

    print(f"[{split}] kept {kept} images -> {out_img}")

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    for sp in args.splits:
        convert_split(sp, root, out, args.copy, args.keep_empty)

    # Write a minimal data.yaml for Ultralytics (HBB)
    data_yaml = f"""# auto-generated for Ultralytics YOLO (HBB)
path: {out}
train: train/images
val: val/images
# test: test-dev/images  # enable if you have GT for test-dev
names: {VISDRONE_CLASSES}
"""
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"[DONE] data.yaml -> {out/'data.yaml'}")

if __name__ == "__main__":
    main()
