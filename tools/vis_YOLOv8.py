from ultralytics import YOLO

model = YOLO("path/to/your/model's/weights/best.pt")
results = model.predict(
    source="path/to/your/images.png",
    conf=0.35,
    save=True,
    # if you don't need text files, turn these off:
    save_txt=False,
    save_conf=False,

    # visualization controls:
    show_labels=False,    # hide class names
    show_conf=False,      # hide confidence text
    line_width=3,    # thinner boxes (try 1 or 2)

    project="vis_yolov8",
    name="method",
    device=0
)
