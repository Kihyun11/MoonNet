from mmdet.apis import init_detector
from inference_YOLC import inference_detector, inference_detector_with_LSM
import numpy as np
import cv2
import mmcv
import os
import json
from mmdet.datasets import build_dataset

from pycocotools.coco import COCO

from models import *
from VisDrone_Dataset import VisDroneDataset


#visual = False
visual = True

# ===== New: global viz settings =====
score_thr = 0.30           # draw only if score >= this
out_dir = "pred_vis"       # where to save drawn images
os.makedirs(out_dir, exist_ok=True)

# a simple but distinct color palette for up to 10 classes (BGR for OpenCV)
PALETTE = [
    (255,  56,  56),  # class 0
    (255, 157, 151),  # 1
    (255, 112,  31),  # 2
    (255, 178,  29),  # 3
    (207, 210,  49),  # 4
    ( 72, 249,  10),  # 5
    (146, 204,  23),  # 6
    ( 61, 219, 134),  # 7
    ( 26, 147,  52),  # 8
    (  0, 204, 255),  # 9
]



def draw_and_save(img_path, img_name, final_result, classes):
    """
    Draw predicted boxes on the image and save to out_dir/img_name
    final_result: list of length num_classes; each is Nx5 array [x1,y1,x2,y2,score]
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path} for visualization.")
        return

    h, w = img.shape[:2]

    for cls_idx, dets in enumerate(final_result):
        if dets is None or len(dets) == 0:
            continue

        # dets is expected to be Nx5
        for det in dets:
            if det is None or len(det) < 5:
                continue
            x1, y1, x2, y2, score = det.astype(float)

            # skip placeholders like [0,0,0,0,0]
            if score <= 0:
                continue
            if score < score_thr:
                continue

            # clip + cast
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            color = PALETTE[cls_idx % len(PALETTE)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{classes[cls_idx]} {score:.2f}"
            # text background
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            th = th + baseline
            y_text = max(0, y1 - 4)
            cv2.rectangle(img, (x1, y_text - th), (x1 + tw + 2, y_text), color, -1)
            cv2.putText(img, label, (x1 + 1, y_text - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    save_path = os.path.join(out_dir, img_name)
    cv2.imwrite(save_path, img)

def inference(model1, model2, img, img_name, num_cls = 10, saved_crop = 2, crop=True):
    # data = dict(img=img)
    subregion_coord, coarse_result = inference_detector_with_LSM(model1, img)

    img_np = cv2.imread(img)
    img_draw = img_np.copy()
    final_result = []
    cluster_regions = []
    
    for i in range(num_cls):
        final_result.append([])

    if crop:
        areas = []
        for item in subregion_coord:
            x,y,w,h = item
            areas.append(w*h)
        areas = np.array(areas)
        idx = areas.argsort()[::-1]
        if len(idx) > saved_crop:
            idx = idx[:saved_crop]

        subregion_coord = subregion_coord[idx]

        for i in range(len(subregion_coord)):
            x, y, w, h = subregion_coord[i]
            # filter small crops
            if w*h < 96*96:
                continue
            x = int(x)
            y = int(y)

            bboxes = [x, y, x+w, y+h]
            box_scale_ratio = 1.2         # box scale factor

            w_half = (bboxes[2] - bboxes[0]) * 0.5
            h_half = (bboxes[3] - bboxes[1]) * 0.5
            x_center = (bboxes[2] + bboxes[0]) * 0.5
            y_center = (bboxes[3] + bboxes[1]) * 0.5

            w_half *= box_scale_ratio
            h_half *= box_scale_ratio
            w, h = img_np.shape[1], img_np.shape[0]

            # scale dense region by 1.2x to avoid truncation
            boxes_scaled = [0, 0, 0, 0]
            boxes_scaled[0] = min(max(x_center - w_half, 0), w - 1)
            boxes_scaled[2] = min(max(x_center + w_half, 0), w - 1)
            boxes_scaled[1] = min(max(y_center - h_half, 0), h - 1)
            boxes_scaled[3] = min(max(y_center + h_half, 0), h - 1)
            cluster_regions.append(boxes_scaled)
            boxes_scaled = [int(i) for i in boxes_scaled]


            img_scale_ratio = 1.5   # image scale factor
            w_new = boxes_scaled[2] - boxes_scaled[0]
            h_new = boxes_scaled[3] - boxes_scaled[1]
            # crop and resize sub-image by 1.5x
            img_crop = img_np[boxes_scaled[1]:boxes_scaled[3], boxes_scaled[0]:boxes_scaled[2]]
            if visual:
                cv2.rectangle(img_draw, (boxes_scaled[0], boxes_scaled[1]), (boxes_scaled[2], boxes_scaled[3]), (0, 0, 0), 2)
            img_resize = cv2.resize(img_crop, (int(w_new * img_scale_ratio), int(h_new * img_scale_ratio)))
        
            result_refine = inference_detector(model2, img_resize)
        
            for i in range(len(coarse_result)):
                final_result[i].append([0,0,0,0,0]) # avoid empty list
                for item in result_refine[i]:
                    item[0:4] = item[0:4] / img_scale_ratio
                    item[0:2] += boxes_scaled[0:2]
                    item[2:4] += boxes_scaled[0:2]
                    final_result[i].append(item)

    # fuse coarse and refined results
    for i in range(len(coarse_result)):
        cls_result = coarse_result[i]
        for item in cls_result:
            x1, y1, x2, y2, score = item
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            in_cluster = False
            # YOLC replaces the coarse results with refined results in cluster regions 
            for boxes_scaled in cluster_regions:
                if boxes_scaled[0] <= x1 and x2 <= boxes_scaled[2] and boxes_scaled[1] <= y1 and y2 <= boxes_scaled[3]:
                    in_cluster = True
                    break
            if not in_cluster:
                final_result[i].append(item)
    
    if visual:
        path = "tmp/"+img.split("/")[-1]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img_draw)

    for i in range(len(final_result)):
        final_result[i] = np.array(final_result[i])

    # ===== New: draw and save this image's predictions =====
    draw_and_save(img, img_name, final_result, classes)

    return final_result




if __name__ == '__main__':
    dataset_anno = '/workspace/visdrone4YOLC/Visdrone/VisDrone2019-DET-val/VisDrone2019-DET-val/valid.json'
    dataset_root = '/workspace/visdrone4YOLC/Visdrone/VisDrone2019-DET-val/VisDrone2019-DET-val/images'

    classes = ('pedestrian', "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")

    config_file1 = '/workspace/YOLC/configs/yolc.py'
    config_file2 = '/workspace/YOLC/configs/yolc.py'

    #config_file1 = '/workspace/YOLC/configs/yolc_moonnet.py'
    #config_file2 = '/workspace/YOLC/configs/yolc_moonnet.py'

    checkpoint_file = '/workspace/YOLC/work_dir/yolc_hrnet/latest.pth'
    #checkpoint_file = '/workspace/YOLC/work_dir/yolc_moonnet_real/latest.pth'

    # Global Image Detector
    model1 = init_detector(config_file1, checkpoint_file, device='cuda:0')
    model1.eval()

    # Crop Image Detector (weight sharing)
    model2 = model1
    # model2 = init_detector(config_file2, checkpoint_file, device='cuda:1')
    # model2.eval()

    saved_crop = 1      # crop region numbers generated by LSM (e.g. k=2)

    with open(dataset_anno) as f:
        json_info = json.load(f)
    annotation_set = {}
    for annotation in json_info['annotations']:
        image_id = annotation['image_id']
        if not image_id in annotation_set.keys():
            annotation_set[image_id] = []
        annotation_set[image_id].append(annotation)

    coco = COCO(dataset_anno)   # Load Val Dataset
    size = len(list(coco.imgs.keys()))  #  Image num
    results = []
    prog_bar = mmcv.ProgressBar(size)
    for key in range(size):
        ids = list(coco.imgs.keys())
        key = ids[key]

        width = coco.imgs[key]['width']
        height = coco.imgs[key]['height']
        img_name = coco.imgs[key]['file_name']
        img = os.path.join(dataset_root, img_name)
        final_result = inference(model1, model2, img, img_name, num_cls=len(classes), saved_crop = 2)
        results.append(final_result)

        prog_bar.update()
    

    eval_kwargs = dict(interval=1, metric='bbox')
    kwargs = {}
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox', **kwargs))
    test_config = dict(
        type='VisDroneDataset',
        classes=classes,
        ann_file=dataset_anno,
        img_prefix=dataset_root,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                #scale_factor=[1.0, 1.25, 1.5],
                #flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ])
    
    dataset = build_dataset(test_config)
    metric = dataset.evaluate(results, **eval_kwargs)
    print(metric)
