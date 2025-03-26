# Safety-Helmet-Detector

## About the Dataset

The dataset can be downloaded from https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/14
You will need a Roboflow account and API key associated with your account to download this dataset. 

## Objective

The purpose of this project:
1. Fine tune a pre-trained object detection model (YOLOv11) and train on this dataset. For more information on YOLOv11 and other models and Roboflow, you can check out this link, https://github.com/roboflow/notebooks 
2. Develop a web app that detects whether a person(s) are wearing safety helmets given a single image or video, using the trained model. 

## Program

The programs were developed in Python. Model training was performed in Google Colab. The app section comprise two python files - a class called HelmetDetector and a main file to run the application. 

## Techniques

   - Deep Learning 
   - Web app development 

## Algorithms 

   - YOLO (Object Detection) 

## Libraries
  
   - Ultralytics
   - Roboflow
   - PyTorch
   - Glob
   - NumPy
   - OpenCV
   - Gradio
   