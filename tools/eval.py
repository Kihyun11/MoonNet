from ultralytics import YOLO

m = YOLO("/path/to/your/backbone.yaml/file")


metrics = m.val(data="/path/to/your/data.yaml/file",
                split="val", 
                imgsz=640, conf=0.001, iou=0.7, save_json=True)

print(metrics.results_dict) 