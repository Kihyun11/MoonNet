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


### Parameter and backbone setting
For the enhancement of the tiny object detection, following python files and yaml file from YOLOv8 had been modified:
1) User/ultralytics/cfg/models/v8/yolov8.yaml
2) User/ultralytics/nn/tasks.py
3) User/ultralytics/nn/modules/conv.py
4) User/ultralytics/nn/modules/init.py
5) User/ultralytics/utils/loss.py
   
Be careful, these directories are partially right up to the folder named ultralytics. The exact directories will vary based on the installation.
Once the directories are sorted, such files should be replaced with the corresponding files in the repository in order to set the exact environment as this project.
yolov8.yaml file updates the new backbone, (tasks.py, conv.py, init.py) updates the two new module called "conv_SEBlock" and "conv_CBAM" and finally loss.py updates new gamma value for focal loss function.
