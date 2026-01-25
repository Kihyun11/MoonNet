# MoonNet
Enhanced Detection of Tiny Objects in Aerial Images

## Dataset
<p align="center">
  <img src="figures/DOTA_modification.png" alt="dota_modification" width="350" height = "350px"/>
  <br>
  <em>Modified DOTAv2.0</em>
</p>

## Backbone Designs
<p align="center">
  <img src="figures/backbone_design.jpg" alt="backbone_design" width="400" height = "350px"/>
  <br>
  <em>Six attention-augmented backbone designs</em>
</p>

## Mixture of Orthogonal Neural Network (MoonNet)
<p align="center">
  <img src="figures/MoonNet.jpg" alt="MoonNet" width="300" height = "350px"/>
  <br>
  <em>MoonNet backbone design</em>
</p>

## Dataset Preparation
Download and prepare the original DOTAv2.0 dataset same as the below folder hierarchy and also create the output folder where the modified dataset will be stored.
```
DOTAv2.0(Original)/
├── train/
│ ├── images/
│ | ├── P0000.png
| | | ...
│ └── labels/
│ | ├── P0000.txt 
| | | ...
|
└── val/
│ ├── images/
│ | ├── P0003.png
| | | ...
│ └── labels/
│ | ├── P0003.txt 
| | | ...
```
Then run this python code
```python
python dota2COCO.py \
--dota_root /path/to/DOTA-v2.0(Original) \
--out /path/to/modified_dota_v2(Output) \
#for HBB extraction,
--export_coco
#for OBB extraction,
--obb
```

## Training
Training using pretrained model
```python
tools/python screening_train.py
```
Training using attention-augmented backbone (training from scratch)
```python
tools/python custom_backbones_train.py
```

## Evaluating
Evaluating using modified DOTA
```python
tools/python eval.py
```
Evaluating using VisDrone
```python
tools/python eval_visdrone.py
```
Visual predictions
```python
tools/python vis_YOLOv8.py
```
