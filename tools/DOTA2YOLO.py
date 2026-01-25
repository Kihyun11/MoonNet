# #!/usr/bin/env python3
# # dota2_to_yolo_filtered.py
# # Python 3.9+
# # Convert DOTA-v2.0 train/val to filtered YOLO dataset with desired folder layout:
# # <out>/{train,val}/{images,labels}/...

# import argparse, json, os, shutil
# from pathlib import Path
# from typing import List, Dict
# from PIL import Image

# # ----- Config -----
# ALLOWED_RAW = {"small-vehicle", "large-vehicle", "plane", "ship", "storage-tank"}
# REMAP = {
#     "small-vehicle": "small_vehicle",
#     "large-vehicle": "large_vehicle",
#     "plane": "plane",
#     "ship": "ship",
#     "storage-tank": "storage_tank",
# }
# # Fixed, deterministic class order
# CLASSES = [REMAP[c] for c in sorted(ALLOWED_RAW)]
# CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
# IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# def parse_args():
#     ap = argparse.ArgumentParser(
#         description="Filter DOTA-v2.0 (train/val) to 5 classes and export YOLO (HBB or OBB)."
#     )
#     ap.add_argument("--dota_root", required=True,
#                     help="Path to DOTA-v2.0 root containing train/ and val/")
#     ap.add_argument("--out", required=True, help="Output directory")
#     ap.add_argument("--copy", action="store_true",
#                     help="Copy images (default tries symlink, falls back to copy)")
#     ap.add_argument("--keep_empty", action="store_true",
#                     help="Keep images that become empty after filtering")
#     ap.add_argument("--export_coco", action="store_true",
#                     help="Also write COCO JSONs (HBB only)")
#     ap.add_argument("--obb", action="store_true",
#                     help="Write YOLO-OBB labels (8 coords) instead of HBB")
#     return ap.parse_args()

# def read_labeltxt(txt_path: Path) -> List[Dict]:
#     """DOTA labelTxt line: x1 y1 x2 y2 x3 y3 x4 y4 cls diff."""
#     objs = []
#     if not txt_path.exists():
#         return objs
#     for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
#         parts = line.strip().split()
#         if len(parts) < 10:
#             continue
#         try:
#             poly = list(map(float, parts[:8]))
#         except ValueError:
#             continue
#         cls = parts[8]
#         objs.append({"poly": poly, "cls": cls})
#     return objs

# def filter_objs(objs: List[Dict]) -> List[Dict]:
#     kept = []
#     for o in objs:
#         if o["cls"] in ALLOWED_RAW:
#             kept.append({"poly": o["poly"], "cls": REMAP[o["cls"]]})
#     return kept

# def write_yolo_hbb(txt_path: Path, objs: List[Dict], w: int, h: int):
#     """YOLO HBB: cls xc yc w h (normalized)."""
#     lines = []
#     for o in objs:
#         xs, ys = o["poly"][0::2], o["poly"][1::2]
#         xmin, xmax = min(xs), max(xs)
#         ymin, ymax = min(ys), max(ys)
#         bw, bh = xmax - xmin, ymax - ymin
#         if bw <= 0 or bh <= 0:
#             continue
#         xc = (xmin + xmax) / 2.0 / w
#         yc = (ymin + ymax) / 2.0 / h
#         bw /= w
#         bh /= h
#         cid = CLASS_TO_ID[o["cls"]]
#         lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
#     txt_path.parent.mkdir(parents=True, exist_ok=True)
#     txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

# def write_yolo_obb(txt_path: Path, objs: List[Dict], w: int, h: int):
#     """YOLO OBB: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
#     lines = []
#     for o in objs:
#         poly = o["poly"]  # [x1,y1,x2,y2,x3,y3,x4,y4]
#         xs = [poly[i] / w for i in range(0, 8, 2)]
#         ys = [poly[i] / h for i in range(1, 8, 2)]
#         if any(v < 0 or v > 1 for v in xs + ys):
#             continue
#         cid = CLASS_TO_ID[o["cls"]]
#         coords = " ".join(f"{v:.6f}" for pair in zip(xs, ys) for v in pair)
#         lines.append(f"{cid} {coords}")
#     txt_path.parent.mkdir(parents=True, exist_ok=True)
#     txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

# def link_or_copy(src: Path, dst: Path, force_copy: bool):
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     if dst.exists():
#         return
#     if force_copy:
#         shutil.copy2(src, dst)
#     else:
#         try:
#             os.symlink(src, dst)
#         except Exception:
#             shutil.copy2(src, dst)

# def process_split(split: str, root: Path, out: Path, copy_flag: bool,
#                   keep_empty: bool, obb: bool) -> List[Path]:
#     """Convert one split (train or val). Returns list of kept image paths in out."""
#     src_img_dir = root / split / "images"
#     src_lab_dir = (root / split / "labelTxt") if (root / split / "labelTxt").exists() else (root / split / "labels")

#     kept_imgs = []

#     # NEW LAYOUT: <out>/<split>/{images,labels}
#     out_img_dir = out / split / "images"
#     out_lab_dir = out / split / "labels"
#     out_img_dir.mkdir(parents=True, exist_ok=True)
#     out_lab_dir.mkdir(parents=True, exist_ok=True)

#     for img in sorted(src_img_dir.iterdir()):
#         if img.suffix.lower() not in IMG_EXTS:
#             continue
#         lab = src_lab_dir / (img.stem + ".txt")
#         objs = read_labeltxt(lab)
#         fobjs = filter_objs(objs)
#         if not fobjs and not keep_empty:
#             continue

#         # place image
#         dst_img = out_img_dir / img.name
#         link_or_copy(img, dst_img, copy_flag)

#         # write label (empty OK if keep_empty=True)
#         with Image.open(dst_img) as im:
#             w, h = im.size
#         yolo_txt = out_lab_dir / f"{img.stem}.txt"
#         if obb:
#             write_yolo_obb(yolo_txt, fobjs, w, h)
#         else:
#             write_yolo_hbb(yolo_txt, fobjs, w, h)

#         kept_imgs.append(dst_img)

#     # optional manifest
#     (out / "splits").mkdir(parents=True, exist_ok=True)
#     (out / "splits" / f"{split}.txt").write_text(
#         "\n".join(p.name for p in kept_imgs) + ("\n" if kept_imgs else ""), encoding="utf-8"
#     )
#     return kept_imgs

# def export_coco_for_split(split: str, out: Path):
#     """Create minimal COCO JSON from YOLO-HBB labels (HBB only)."""
#     ann_dir = out / "annotations"
#     ann_dir.mkdir(parents=True, exist_ok=True)

#     # NEW LAYOUT
#     img_dir = out / split / "images"
#     lab_dir = out / split / "labels"

#     images, annotations = [], []
#     ann_id = 1
#     idx = 1
#     for img in sorted(img_dir.iterdir()):
#         if img.suffix.lower() not in IMG_EXTS:
#             continue
#         with Image.open(img) as im:
#             w, h = im.size
#         images.append({
#             "id": idx,
#             "file_name": img.relative_to(out).as_posix(),
#             "width": w,
#             "height": h
#         })

#         ytxt = lab_dir / f"{img.stem}.txt"
#         if ytxt.exists():
#             for line in ytxt.read_text().splitlines():
#                 parts = line.strip().split()
#                 if len(parts) != 5:
#                     continue
#                 cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
#                 x = (xc - bw / 2) * w
#                 y = (yc - bh / 2) * h
#                 ww = bw * w
#                 hh = bh * h
#                 annotations.append({
#                     "id": ann_id,
#                     "image_id": idx,
#                     "category_id": cid,
#                     "bbox": [x, y, ww, hh],
#                     "area": ww * hh,
#                     "iscrowd": 0,
#                     "segmentation": []
#                 })
#                 ann_id += 1
#         idx += 1

#     categories = [{"id": CLASS_TO_ID[n], "name": n, "supercategory": "object"} for n in CLASSES]
#     coco = {"images": images, "annotations": annotations, "categories": categories}
#     (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")

# def main():
#     args = parse_args()
#     root = Path(args.dota_root)
#     out = Path(args.out)
#     out.mkdir(parents=True, exist_ok=True)

#     kept_train = process_split("train", root, out, args.copy, args.keep_empty, args.obb)
#     kept_val   = process_split("val",   root, out, args.copy, args.keep_empty, args.obb)

#     # data.yaml (Ultralytics)
#     # IMPORTANT: paths now point to <out>/train/images and <out>/val/images
#     task_line = "task: obb\n" if args.obb else ""
#     data_yaml = f"""# auto-generated
# {task_line}path: {out.resolve()}
# train: train/images
# val: val/images
# names: {CLASSES}
# """
#     (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

#     if args.export_coco and not args.obb:
#         export_coco_for_split("train", out)
#         export_coco_for_split("val", out)

#     print(f"[DONE] Kept images -> train={len(kept_train)}  val={len(kept_val)}")
#     print("OBB labels written." if args.obb else "HBB labels written.")
#     print(f"Ultralytics data.yaml: {out/'data.yaml'}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# Python 3.9+
# Convert DOTA-v2.0 (train/val) to filtered YOLO dataset (HBB or OBB) with a fixed 5-class subset.
# Layout:
#   <out>/
#     train/{images,labels}
#     val/{images,labels}
#     splits/
#     data.yaml
#     annotations/ (only if --export_coco and not --obb)

import argparse, json, os, shutil
from pathlib import Path
from typing import List, Dict
from PIL import Image
import math

# ----- Config -----
ALLOWED_RAW = {"small-vehicle", "large-vehicle", "plane", "ship", "storage-tank"}
REMAP = {
    "small-vehicle": "small_vehicle",
    "large-vehicle": "large_vehicle",
    "plane": "plane",
    "ship": "ship",
    "storage-tank": "storage_tank",
}
# Fixed, deterministic class order (alphabetical on raw names, then remapped)
CLASSES = [REMAP[c] for c in sorted(ALLOWED_RAW)]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter DOTA-v2.0 (train/val) to 5 classes and export YOLO (HBB or OBB)."
    )
    ap.add_argument("--dota_root", required=True,
                    help="Path to DOTA-v2.0 root containing train/ and val/")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--copy", action="store_true",
                    help="Copy images (default tries symlink, falls back to copy)")
    ap.add_argument("--keep_empty", action="store_true",
                    help="Keep images that become empty after filtering")
    ap.add_argument("--export_coco", action="store_true",
                    help="Also write COCO JSONs (HBB only)")
    ap.add_argument("--obb", action="store_true",
                    help="Write YOLO-OBB labels (8 coords) instead of HBB")
    return ap.parse_args()

# -----------------------------
# DOTA parsing and filtering
# -----------------------------

def read_labeltxt(txt_path: Path) -> List[Dict]:
    """
    DOTA labelTxt expected line:
        x1 y1 x2 y2 x3 y3 x4 y4 cls diff
    We parse the first 8 floats as polygon, the 9th token as class.
    """
    objs = []
    if not txt_path.exists():
        return objs
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        try:
            poly = list(map(float, parts[:8]))
        except ValueError:
            continue
        cls = parts[8]
        objs.append({"poly": poly, "cls": cls})
    return objs

def filter_objs(objs: List[Dict]) -> List[Dict]:
    kept = []
    for o in objs:
        if o["cls"] in ALLOWED_RAW:
            kept.append({"poly": o["poly"], "cls": REMAP[o["cls"]]})
    return kept

# -----------------------------
# OBB canonicalization helpers
# -----------------------------

def _signed_area(xs, ys):
    # Shoelace; positive => CCW, negative => CW
    s = 0.0
    for i in range(4):
        j = (i + 1) % 4
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return 0.5 * s

def _to_clockwise(xs, ys):
    # If CCW, reverse to CW (but keep it cyclic)
    if _signed_area(xs, ys) > 0:  # CCW
        xs = [xs[0], xs[3], xs[2], xs[1]]
        ys = [ys[0], ys[3], ys[2], ys[1]]
    return xs, ys

def _start_from_topleft(xs, ys):
    # Choose the vertex with minimal (x+y) as the start, rotate cyclically
    scores = [xs[i] + ys[i] for i in range(4)]
    k = min(range(4), key=lambda i: scores[i])
    xs = xs[k:] + xs[:k]
    ys = ys[k:] + ys[:k]
    return xs, ys

# -----------------------------
# Writers (no clipping/dropping)
# -----------------------------

def write_yolo_hbb(txt_path: Path, objs: List[Dict], w: int, h: int):
    """
    YOLO HBB format: cls xc yc w h (normalized).
    We DO NOT clip or drop out-of-bounds; values may be <0 or >1 if polys cross the border.
    """
    lines = []
    for o in objs:
        xs, ys = o["poly"][0::2], o["poly"][1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 0 or bh <= 0:
            # degenerate, skip (only true zero-width/height)
            continue
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        bw /= w
        bh /= h
        cid = CLASS_TO_ID[o["cls"]]
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def write_yolo_obb(txt_path: Path, objs: List[Dict], w: int, h: int):
    """
    YOLO OBB format: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized, CW, start=top-left).
    We DO NOT clip or drop out-of-bounds; we only canonicalize orientation and start point.
    """
    lines = []
    for o in objs:
        poly = o["poly"]  # [x1,y1,x2,y2,x3,y3,x4,y4] in pixels
        xs = [poly[i] / w for i in range(0, 8, 2)]
        ys = [poly[i] / h for i in range(1, 8, 2)]

        # enforce clockwise orientation and canonical start, no clipping
        xs, ys = _to_clockwise(xs, ys)
        xs, ys = _start_from_topleft(xs, ys)

        # simple degeneracy guard: reject strictly zero area
        if abs(_signed_area(xs, ys)) == 0.0:
            continue

        cid = CLASS_TO_ID[o["cls"]]
        coords = " ".join(f"{v:.6f}" for pair in zip(xs, ys) for v in pair)
        lines.append(f"{cid} {coords}")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

# -----------------------------
# IO helpers and per-split work
# -----------------------------

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

def process_split(split: str, root: Path, out: Path, copy_flag: bool,
                  keep_empty: bool, obb: bool) -> List[Path]:
    """Convert one split (train or val). Returns list of kept image paths in out."""
    src_img_dir = root / split / "images"
    src_lab_dir = (root / split / "labelTxt") if (root / split / "labelTxt").exists() else (root / split / "labels")

    kept_imgs = []

    # Target layout: <out>/<split>/{images,labels}
    out_img_dir = out / split / "images"
    out_lab_dir = out / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    for img in sorted(src_img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lab = src_lab_dir / (img.stem + ".txt")
        objs = read_labeltxt(lab)
        fobjs = filter_objs(objs)
        if not fobjs and not keep_empty:
            continue

        # place image
        dst_img = out_img_dir / img.name
        link_or_copy(img, dst_img, copy_flag)

        # write label
        with Image.open(dst_img) as im:
            w, h = im.size
        yolo_txt = out_lab_dir / f"{img.stem}.txt"
        if obb:
            write_yolo_obb(yolo_txt, fobjs, w, h)
        else:
            write_yolo_hbb(yolo_txt, fobjs, w, h)

        kept_imgs.append(dst_img)

    # optional manifest for convenience
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / f"{split}.txt").write_text(
        "\n".join(p.name for p in kept_imgs) + ("\n" if kept_imgs else ""), encoding="utf-8"
    )
    return kept_imgs

# -----------------------------
# COCO (HBB only) export
# -----------------------------

def export_coco_for_split(split: str, out: Path):
    """Create minimal COCO JSON from YOLO-HBB labels (HBB only)."""
    ann_dir = out / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out / split / "images"
    lab_dir = out / split / "labels"

    images, annotations = [], []
    ann_id = 1
    idx = 1
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        with Image.open(img) as im:
            w, h = im.size
        images.append({
            "id": idx,
            "file_name": img.relative_to(out).as_posix(),
            "width": w,
            "height": h
        })

        ytxt = lab_dir / f"{img.stem}.txt"
        if ytxt.exists():
            for line in ytxt.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                x = (xc - bw / 2) * w
                y = (yc - bh / 2) * h
                ww = bw * w
                hh = bh * h
                annotations.append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": cid,
                    "bbox": [x, y, ww, hh],
                    "area": ww * hh,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
        idx += 1

    categories = [{"id": CLASS_TO_ID[n], "name": n, "supercategory": "object"} for n in CLASSES]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")

# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    root = Path(args.dota_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept_train = process_split("train", root, out, args.copy, args.keep_empty, args.obb)
    kept_val   = process_split("val",   root, out, args.copy, args.keep_empty, args.obb)

    # data.yaml (Ultralytics)
    # IMPORTANT: paths point to <out>/train/images and <out>/val/images
    task_line = "task: obb\n" if args.obb else ""
    data_yaml = f"""# auto-generated
{task_line}path: {out.resolve()}
train: train/images
val: val/images
names: {CLASSES}
"""
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    if args.export_coco and not args.obb:
        export_coco_for_split("train", out)
        export_coco_for_split("val", out)

    print(f"[DONE] Kept images -> train={len(kept_train)}  val={len(kept_val)}")
    print("OBB labels written." if args.obb else "HBB labels written.")
    print(f"Ultralytics data.yaml: {out/'data.yaml'}")

if __name__ == "__main__":
    main()
