# Mixture of Orthogonal Neural-modules Network(MoonNet): Enhanced-Detection-of-Tiny-Objects-Using-Machine-Learning-Algorithm
Imperial College London MEng EIE Final Year Project
### About
The objective of this project is to be able to enhance the detection of tiny objects in the [DOTA](https://captain-whu.github.io/DOTA/dataset.html)(Dataset for Object detection in Aerial images) using [YOLOv8](https://github.com/ultralytics/ultralytics) which is the latest version of the YOLO(You Only Look Once) model.

Following approaches were taken for enhancement:
1) Hyper parameter tuning for optimal training - (focusing parameter value modification for focal loss, optimizer selection and varying resolution)
2) Data augmentation
3) Multi-scale feature learning
4) Attnetion mechanism - (Implementation of [SE Block](https://github.com/hujie-frank/SENet?tab=readme-ov-file) and [CBAM](https://arxiv.org/abs/1807.06521) in the backbone of YOLOv8 model)

### Data Explanation
For this project, DOTA-v2.0 had been used and few modifications were made before training.
1) The original dataset, containing 18 categories, was streamlined to include only 5 targeted categories: large vehicles, small vehicles, ships, planes, and storage tanks. This refinement aimed to sharpen the focus on the detection of smaller objects, as the expansive range of categories in the original set included items not relevant to the project's scope.
2) The dataset's format was transformed to be compatible with the YOLO framework, resulting in the creation of a corresponding YAML configuration file. 
3) Various data augmentation techniques were applied to enhance detection accuracy and elevate the model's performance.

### YOLOv8 Installation
```python
pip install ultralytics==8.3.162
```
Alternative installation step and more detailed information about the installation can be found here [YOLOv8](https://github.com/ultralytics/ultralytics). It is beneficial to install ultralytics version 8.3.162 which is the exact environment of this project.

### Python Library Installation
Following python libraries are required before starting the training
```python
pip install opencv-python
```
```python
pip install pip install matplotlib
```
```python
pip install pandas
```
```python
pip install scipy
```
```python
pip install tqdm
```

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
yolov8.yaml file updates the new backbone, (tasks.py, conv.py, init.py) updates the two new module called "conv_SEBlock" and "conv_CBAM" and finally loss.py updates new gamma value for focal loss function. If you installed different version of ultralytics other than 8.0.222, you need to manually modify the above files in order to set the exact environment. This is manually guided in the next sections, however, manual changes might not work for the latest version. In that case make sure you install the version 8.0.222.

### 1) Hyper parameter tuning (Manual setting)
Image size setting - In this project, image size of 928 is selected for the final training and thus imgsz parameter is set to 928
```python
from ultralytics import YOLO
results = model.train(data=data_path, epochs=12, imgsz=928)
```

Optimizer setting - In this project, the SGD optimizer is selected for the optimal training and thus used for the final training. Optimizer can be easily selected by editing the default.yaml file under ultralytics/cfg folders
```yaml
optimizer: SGD # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
```

Focusing parameter (Gamma) value setting for verifocal and focal loss - The default setting of focusing parameter values for the verifocal loss and focal loss are 2.0 (verifocal) and 1.5 (focal). In this project these values are modifed to 5.0 and 4.5. Such modification can be made on the loss.py file under ultralytics/utils folders
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

### 2) Data Augmentation
In this project, data augmentation version 2 is selected for the final training. More various data augmentation techniques can be implemented on the data augmentation version 1 which is the data pack with no augmentation.
When using the dataset, we have to specify the directory for the data.yaml file in the dataset. In the uploaded dataset files, data.yaml file exists for the each dataset file.

First, we need to modify this data.yaml file. We need to specify the directories for the training set, validation set and the test set.

```yaml
names:
- large-vehicle
- plane
- ship
- small-vehicle
- storage-tank
nc: 5
roboflow:
...
test: Users/test/images # This is your directory for the test set
train: Users/train/images # This is your directory for the train set
val: Users/valid/images # This is your directory for the validation set
```
Once this data.yaml file is correctly modified. We can use this dataset in the training.
We can specify the directory of this modified yaml file before training.

```python
results = model.train(data='Users/data.yaml', epochs=12, imgsz=928) # You need to specify your directory for data.yaml file here
```

### 3) Multi scale feature learning and Attention Mechanism (Manual setting)
For the multi scale feature learning strategy used in this project, dilated layers with dilation factor 2 and 3 are used. Thus, when manually setting the model, two dilated layers need to be added.

The first step is adding two classes 'Dilated_Conv' and 'Dilated_Conv_d3' in conv.py file under ultralytics/nn/modules folders.

```python
class Dilated_Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    # d = 2 is the only difference

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=2, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
```
```python
class Dilated_Conv_d3(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    # d = 2 is the only difference

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=3, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
```

Since the two dilated layers are added, now it's time to add two attention modules. For the attention mechanism strategy used in this project, the two attention modules SE BLock and CBAM are used. Thus, these two modules need to be added as well.
The second step is adding two classes 'Conv_SEBlock' and 'Conv_CBAM' in conv.py file under ultralytics/nn/modules folders.

```python
class Conv_SEBlock(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        #--------------------------------------------------------------------
        # Squeeze-and-Excitation layers
        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(c2, c2 // 16, 1)
        self.se_fc2 = nn.Conv2d(c2 // 16, c2, 1)
        self.se_act = nn.Sigmoid()
        #--------------------------------------------------------------------
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        #return self.act(self.bn(self.conv(x)))
        #-------------------------------------
        x = self.act(self.bn(self.conv(x)))
        # Squeeze-and-Excitation operations
        y = self.se_squeeze(x)
        y = self.se_fc1(y)
        y = F.relu(y, inplace=True)
        y = self.se_fc2(y)
        y = self.se_act(y)
        x = x * y
        return x

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        #return self.act(self.conv(x))
        #-------------------------------------
        x = self.act(self.conv(x))

         # Squeeze-and-Excitation operations
        y = self.se_squeeze(x)
        y = self.se_fc1(y)
        y = F.relu(y, inplace=True)
        y = self.se_fc2(y)
        y = self.se_act(y)
        x = x * y
        #-------------------------------------
        return x
```

```python
class Conv_CBAM(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        #--------------------------------------------------------------------
        #CBAM
        self.channel_attention = ChannelAttention(c2)
        self.spatial_attention = SpatialAttention(7)
        #--------------------------------------------------------------------
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x= self.act(self.bn(self.conv(x)))
        #-------------------------------------
        #return for CBAM
        return self.spatial_attention(self.channel_attention(x))
        #-------------------------------------
        #return x
    def autopad(k, p=None, d=1):
        """Automatic padding calculation."""
        if p is None:  # pad to 'same'
            p = (k - 1) // 2 * d
        return p
```

Once these classes are added correctly, these classes need to be updated.
In the same python file conv.py, add the newly generated classes under \_all_.

```python
__all__ = (
    "Conv",
...
    "Conv_CBAM",
    "Conv_SEBlock",
    "Dilated_Conv",
    "Dilated_Conv_d3",
)
```
and make sure the imports in the file is same as below

```python
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
```
This is the end of the modifications for conv.py file. Next we need to update the '\_init_.py' file under ultralytics/nn/modules folders.
First, update the newly generated classes under \_all_.

```python
__all__ = (
    "Conv",
...
    "Conv_CBAM",
    "Conv_SEBlock",
    "Dilated_Conv",
    "Dilated_Conv_d3",
)
```
Next, update these classes from conv.py file under 'from.conv import'

```python
from .conv import (
    ChannelAttention,
    Concat,
    Conv,
...
    Conv_CBAM,
    Conv_SEBlock,
    Dilated_Conv,
    Dilated_Conv_d3,
)
```
This is the end of the modifications for \_init_.py file. Next we need to update the 'tasks.py'file under ultralytics/nn folders.

In the tasks.py file, we need to update the newly generated classes under 'from ultralytics.nn.modules import'

```python
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
...
    Conv_CBAM,
    Conv_SEBlock,
    Dilated_Conv,
    Dilated_Conv_d3,
)
```
Once the update is done, we need to modify the 'def parse_model(d, ch, verbose=True)' function in the same file.
This file is huge file so careful modification is required. In the version 8.0.222, the parse_model function is located at line 810 of the file.
In the parse_model, following modification is required:
```python
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast
...
if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
            Classify,
            Conv,
...
            Conv_CBAM,
            Conv_SEBlock,
            Dilated_Conv,
            Dilated_Conv_d3,
        ):
...
   return nn.Sequential(*layers), sorted(save)
```
Under 'if verbose' if statement 'if m in' exists and we need to put Conv_CBAM, Conv_SEBlock, Dilated_Conv and Dilated_Conv_d3.
This is the end of modification for tasks.py file.

Now, we can use the newly added classes in the backbone via yaml file.

The backbone yaml file can be found under ultralytics/cfg/models/v8/yolov8.yaml.
If we want to implement Conv_CBAM in the second layer following modifications can update the backbone of the model.

```yaml
# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv_CBAM, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
```
If you want to start training using the customized backbone, the directory of the backbone yaml file need to be specified when begin the training.

```python
from ultralytics import YOLO
model_path = 'Users/customized_backbone.yaml' # This is your directorcy
model = YOLO(model_path)
#model = YOLO(model_path).load('yolov8n.pt')  # build from YAML and transfer weights

results = model.train(data='Users/data.yaml', epochs=12, imgsz=928)
```

### Training and Testing

Once the setting is done, training and testing can be made.
Clear explanations and guide lines for the training using python code are availabe via 'YOLOv8_test.ipynb' file in the repository.

### Results

All the results obtained from this project is stored under the foler called 'results'. The results for different focusing parameter values for the focal loss are stored in the folder called gamma values.


