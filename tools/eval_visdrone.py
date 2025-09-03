from ultralytics import YOLO

m = YOLO("/path/to/your/backbone.yaml/file")


metrics = m.val(data="/path/to/your/data.yaml/file",
                split="val", 
                imgsz=640, conf=0.001, iou=0.7, save_json=True)

#data = /workspace/MoonNet/data/augmentation_v1(no_aug)/data.yaml
#data = /workspace/MoonNet/data/augmentation_v2/data.yaml
print(metrics.results_dict) 