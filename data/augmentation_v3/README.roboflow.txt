
FYP_YOLOV8 - v6 FYP_w_bbox
==============================

This dataset was exported via roboflow.com on February 22, 2024 at 1:45 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3842 images.
Tiny-objects are annotated in YOLOv8 Oriented Object Detection format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 912x912 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random brigthness adjustment of between -50 and 0 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 2.5 pixels

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -20 and +20 degrees
* Random brigthness adjustment of between -30 and 0 percent
* Random exposure adjustment of between -25 and +25 percent


