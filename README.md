# Convolutional Neural Network (CNN) Projects
Convolutional Neural Networks (CNNs) are a class of deep learning models particularly effective for processing structured grid data, such as images. CNNs are designed to automatically learn spatial hierarchies of features from images through layers of convolutional filters, making them highly effective for image classification, object detection, and other visual tasks.


## Projects
## Project 1: [House Price](https://github.com/elenshahbazyan/CNN/blob/main/House%20Price%20Prediction/House%20(1).ipynb) Prediction 
### Description
This project uses a Kaggle dataset to predict house prices based on various features such as the number of rooms, square footage, location, and more. The model applies CNNs in a regression context to predict continuous values like house prices.The project consists of several important steps that need to be followed to achieve the best results:

1. Train the Model for House Price Prediction
We will train a machine learning model to predict house prices using the Kaggle dataset.

2. Preprocess the Dataset
The dataset needs to be preprocessed by cleaning it, handling missing values, and scaling the features as required.

3. Build and Tune the Model Architecture
We need to design and implement a model architecture that is appropriate for the task, ensuring the model has the potential to achieve high accuracy.

4. Optimize Hyperparameters
We will experiment with various hyperparameter combinations to achieve the best possible performance for our model.

5. Evaluate the Model
The model will be evaluated on both training and validation datasets. The evaluation will include:
-Loss graphs (training/validation loss)
-Performance metrics: RMSE (Root Mean Square Error) and R² (Coefficient of Determination)

6. Save the Model
Once the model has been trained and evaluated, we will save it to a file so that it can later be loaded for inference on new data.

### Reference
In this project, we aim to train a model to predict house prices based on a dataset obtained from Kaggle. The dataset is available from the following link: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?resource=download


## Project 2: [CIFAR-10](https://github.com/elenshahbazyan/CNN/blob/main/CIFAR10/CIFAR10.ipynb) Image Classification
### Description
This project uses the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes. The goal is to build a CNN model that can classify images into one of these 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.In this project, we will fine-tune a pre-trained image classification model on the CIFAR-10 dataset. The steps involved in the process are as follows:

1. Fine-Tune the Image Classification Model for CIFAR-10 Dataset
We will fine-tune a pre-trained model to classify images in the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes.

2. Use Pre-Trained Model: ResNet18
We will use the ResNet18 model, which is a deep residual network. Transfer learning will be applied to leverage the pre-trained weights from ResNet18 and fine-tune the model for improved performance on the CIFAR-10 dataset.

3. Design and Build a Model with the Best Architecture
We will experiment with different architectures and fine-tune them to achieve the best accuracy possible. We will also compare the results to established benchmarks and fine-tune the model to reach or exceed these benchmarks.

4. Optimize Hyperparameters
We will test and tune various hyperparameters (e.g., learning rate, batch size, number of epochs) to achieve the best performance for the model.

5. Evaluate the Model
After training, we will evaluate the model on both the training and validation sets.

6.Save the Model
After training and evaluation, we will save the model to a file so that it can later be loaded for inference on new images.

### Reference
The dataset is available from the following source: https://www.cs.toronto.edu/~kriz/cifar.html
## Project 3: [YOLO Training](https://github.com/elenshahbazyan/CNN/blob/main/CIFAR10/CIFAR10.ipynb) Pipeline

An end-to-end object detection training pipeline built on top of Ultralytics YOLOv8, using the Pascal VOC 2012 dataset. This pipeline handles dataset preparation, annotation conversion, hyperparameter tuning with Optuna, model training, evaluation, and metric visualization.

## Project Structure

YOLOTrainingPipeline/
│
├── datasets/
│   └── VOC2012/           # Pascal VOC 2012 dataset (after unzip)
│
├── annotations/           # Converted YOLO-format labels
│
├── runs/                  # YOLOv8 training outputs
│
├── yolov8_pipeline.py     # Main training pipeline class
├── config.yaml            # YOLO training config (auto-generated)
├── README.md              # This file
## Features
-Dataset Loader for Pascal VOC 2012

-Annotation Converter from VOC XML to YOLO TXT format

-Hyperparameter Tuning using Optuna

-Evaluation Metrics using COCO and Pascal VOC mAP

-Loss Curve Visualization with Matplotlib

-Model Saving and easy access to best.pt

## Requirements
Install dependencies:

-pip install -r requirements.txt
-Required packages include:

--ultralytics

--matplotlib

--opencv-python

--scikit-learn

--pandas

--optuna

--seaborn

--pycocotools

## Dataset
Download Pascal VOC 2012 from official site or Kaggle.

Unzip into a folder:

datasets/VOC2012/
🛠️ Usage
python
Копировать
Редактировать
from yolov8_pipeline import YOLOTrainingPipeline

pipeline = YOLOTrainingPipeline(
    dataset_path='datasets/VOC2012',
    yolo_output_dir='runs',
    annotations_output_dir='annotations',
    model_name='yolov8n.pt',
    max_epochs=100,
    patience=20
)

pipeline.prepare_dataset()
pipeline.convert_annotations()
pipeline.create_yolo_config()
pipeline.tune_hyperparameters(n_trials=25)
pipeline.train_model()
pipeline.evaluate_model()
pipeline.plot_metrics()
⚙️ Configuration
A config file like below will be auto-generated:

yaml
Копировать
Редактировать
path: .
train: images/train
val: images/val
test: images/test
nc: 20
names: ['aeroplane', 'bicycle', ..., 'tvmonitor']
📈 Output
After training:

Best model: runs/train/exp/weights/best.pt

Training logs: results.csv

Evaluation metrics printed & saved

Loss and metric plots generated

📊 Evaluation
Supports:

mAP@50 (Pascal VOC)

mAP@50:95 (COCO)

Precision / Recall

📸 Visualization
Training curves:

📉 Box, Class, and DFL Loss

📈 Learning Rate

Example:

(provide image if available)

🧪 Hyperparameter Tuning
Uses Optuna to find optimal:

Learning rate

Momentum

Weight decay

Batch size

Augmentation params
