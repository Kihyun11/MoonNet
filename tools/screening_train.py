from ultralytics import YOLO

# To start the training using pre-trained model, we should select the pre-trained model first.
# In the default setting, 'yolov8n.pt' is used
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# To start the train, you should copy and paste the directory of the YAML file for the dataset into data.
results = model.train(data='/path/to/your/data.yaml/file', 
                      batch = 4, epochs=50, imgsz=640, optimizer ='SGD')