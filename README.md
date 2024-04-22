# Enhanced-Detection-of-Tiny-Objects-Using-Machine-Learning-Algorithm
Imperial College London MEng EIE Final Year Project
### About
The objective of this project is to be able to enhance the detection of tiny objects in the [DOTA](https://captain-whu.github.io/DOTA/dataset.html)(Dataset for Object detection in Aerial images) using [YOLOv8](https://github.com/ultralytics/ultralytics) which is the latest version of the YOLO(You Only Look Once) model.

Following approaches were taken for enhancement:
1) Hyper parameter tuning for optimal training - (gamma value modification for focal loss, optimizer selection and varying resolution)
2) Data augmentation
3) Multi-scale feature learning
4) Contex based strategy - (Implementation of [SE Block](https://github.com/hujie-frank/SENet?tab=readme-ov-file) and [CBAM](https://arxiv.org/abs/1807.06521) in the backbone of YOLOv8 model)

### Data Explanation
For this project, DOTA-v2.0 had been used and few modifications were made before training.
1) The original dataset, containing 18 categories, was streamlined to include only 5 targeted categories: large vehicles, small vehicles, ships, planes, and storage tanks. This refinement aimed to sharpen the focus on the detection of smaller objects, as the expansive range of categories in the original set included items not relevant to the project's scope.
2) The dataset's format was transformed to be compatible with the YOLO framework, resulting in the creation of a corresponding YAML configuration file. 
3) Various data augmentation techniques were applied to enhance detection accuracy and elevate the model's performance.

### YOLOv8 Installation
```python
pip install ultralytics
```
Alternative installation step and more detailed information about the installation can be found here [YOLOv8](https://github.com/ultralytics/ultralytics)


### Parameter and backbone setting
For the enhancement of the tiny object detection, following python files and yaml file from YOLOv8 had been modified:
1) User/ultralytics/cfg/models/v8/yolov8.yaml
2) User/ultralytics/cfg/default.yaml
3) User/ultralytics/nn/tasks.py
4) User/ultralytics/nn/modules/conv.py
5) User/ultralytics/nn/modules/init.py
6) User/ultralytics/utils/loss.py
   
Be careful, these directories are partially right up to the folder named ultralytics. The exact directories will vary based on the installation.
Once the directories are sorted, such files should be replaced with the corresponding files in the repository in order to set the exact environment as this project.
yolov8.yaml file updates the new backbone, (tasks.py, conv.py, init.py) updates the two new module called "conv_SEBlock" and "conv_CBAM" and finally loss.py updates new gamma value for focal loss function.

### 1) Hyper parameter tuning
Image size setting - In this project, image size of 867 was selected for the optimal training and thus imgsz parameter is set to 867
```python
from ultralytics import YOLO
results = model.train(data=data_path, epochs=12, imgsz=867)
```

Optimizer setting - In this project, the SGD optimizer is selected for the optimal training and thus used for the final training. Optimizer can be easily selected by editing the default.yaml file under ultralytics/cfg folders
```yaml
optimizer: SGD # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
```

Gamma parameter setting for verifocal and focal loss - The default setting of gamma parameters for the verifocal loss and focal loss are 2.0 (verifocal) and 1.5 (focal). In this project these values are modifed to 5.0 and 4.5. Such modification can be made on the loss.py file under ultralytics/utils folders
```python
class VarifocalLoss(nn.Module):
...

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    #default vale: alpha = 0.75, gamma = 2.0
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=5.0): #Gamma value modified to 5.0 for verifocal loss
      ...
        return loss

class FocalLoss(nn.Module):
...
    #default vale: alpha = 0.25, gamma = 1.5
    def forward(pred, label, gamma=4.5, alpha=0.25): # Gamma value modified to 4.5 for focal loss
      ...
        return loss.mean(1).sum()
```

