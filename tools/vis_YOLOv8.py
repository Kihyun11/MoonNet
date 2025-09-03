from ultralytics import YOLO

model = YOLO("/path/to/your/model/pth")
results = model.predict(
    source="/path/to/your/test/images",
    conf=0.35,
    save=True,
    # if you don't need text files, turn these off:
    save_txt=False,
    save_conf=False,

    # visualization controls:
    show_labels=False,    # hide class names
    show_conf=False,      # hide confidence text
    line_width=1,    # thinner boxes (try 1 or 2)

    project="vis_yolov8",
    name="model_4",
    device=0
)
