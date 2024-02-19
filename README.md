# Enhanced-Detection-of-Tiny-Objects-Using-Machine-Learning-Algorithm
Imperial College London MEng EIE Final Year Project
### About
The objective of this project is to be able to enhance the detection of tiny objects in the [DOTA](https://captain-whu.github.io/DOTA/dataset.html)(Dataset for Object detection in Aerial images) using [YOLOv8](https://github.com/ultralytics/ultralytics) which is the latest version of the YOLO(You Only Look Once) model.

### Data Explanation
For this project, DOTA-v2.0 had been used and few modifications were made before training.
1) Unlike the original DOTA with 18 categories, it had been modified to have only 5 categories which are [large vehicles, small vehicles, ship, plane, storage tank]. The reason for such     modification is to focus on the detection of tiny objects since not all 18 categories are considered to be tiny.
2) DOTA form has been converted into appropriate form to be trained using YOLO and yaml file had been generated.
3) Various augmentation techniques were implemented in order to increase the detection accuracy and performance.

### YOLOv8 Installation
```bash
pip install ultralytics
```
Alternative installation step and more detailed information about the installation can be found here [YOLOv8](https://github.com/ultralytics/ultralytics)
