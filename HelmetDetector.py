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
        
        
        #Class names for hard hat detection (update these based on your specific model)
        self.class_names = ['helmet']
        
        #Colors for visualization (BGR format)
        self.colors = {
            'helmet': (0, 255, 255),   # Yellow
           
        }
         
    def detect (self, img):
        """Run detection on an image and return annotated image with bounding boxes"""
        #Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Handle gradio input (could be filepath or numpy array)
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for YOLO
        
        # If it's already a numpy array from gradio, ensure it's in the right format
        elif isinstance(img, np.ndarray):
            # If image is BGR (from OpenCV), convert to RGB for YOLO
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        # Make a copy for drawing
        result_img = img.copy()
        
        # Run inference
        results = self.model(img, device)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get confidence score
                conf = float(box.conf[0].cpu().numpy())
                
                # Get class ID
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
                # Draw bounding box
                color = self.colors.get(cls_name, (255, 0, 0))  # Default to blue if class not in colors
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{cls_name}: {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result_img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert back to BGR for OpenCV display if needed
        # result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        
        return result_img
    
    def process_video(self, video_path):
        """Process a video file and return the path to the processed video"""
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file"
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video file
        output_path = video_path.replace('.', '_processed.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            processed_frame_rgb = self.detect(frame_rgb)
            
            # Convert back to BGR for saving with OpenCV
            processed_frame = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Write frame to output video
            out.write(processed_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return output_path
