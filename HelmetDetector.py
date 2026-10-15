import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class HelmetDetector:
    def __init__(self, model_path=None):
        #Load the pretrained YOLO model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            
            
        else:
            
            self.model = YOLO('yolov11n')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        #Class names for hard hat detection (update these based on your specific model)
        self.class_names = ['helmet']
        
        #Colors for visualization (BGR format)
        self.colors = {
            'helmet': (0, 255, 255),   # Yellow
           
        }

    def _predict(self, source, **kwargs):
        return self.model.predict(
            source,
            device=self.device,
            verbose=False,
            **kwargs,
        )
         
    def detect (self, img):
        """Run detection on an image and return annotated image with bounding boxes"""
        
        # Handle gradio input (could be filepath or numpy array)
        if isinstance(img, np.ndarray) and img.ndim == 3:
            source = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            source = img

        results = self._predict(source)
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        return annotated_img
    
    def process_video(self, video_path, frame_stride = 2):
        """Process a video with streamed inference on every Nth frame."""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file"
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        stem, ext = os.path.splitext(video_path)
        output_path = f"{stem}_processed{ext or '.mp4'}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        annotated = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if annotated is None or frame_idx % frame_stride == 0:
                results = self._predict(frame, stream = True)
                for result in results:

                    annotated = result.plot()

            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        
        return output_path
