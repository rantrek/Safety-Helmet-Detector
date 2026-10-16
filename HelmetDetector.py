import os
from time import time
import logging
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

        #Setting up logging handlers
        logging.basicConfig(filename="app.log",
                    format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filemode='w')
        self.logger = logging.getLogger()
        


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
        inference_time = results[0].speed["inference"]
        #self.logger.info("Inference Time: %f ms", inference_time)
        return annotated_img
    
    def process_video(self, video_path, skip_stride = 3):
        """Process a video with streamed inference on every Nth frame."""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file"
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stem, ext = os.path.splitext(video_path)
        output_path = f"{stem}_processed{ext or '.mp4'}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_cc = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter(output_path, fourcc, fps_cc, (width, height))
        
        annotated = None
        frame_idx = 0
        fps_smooth = 0.0
        pipeline_fps = 0.0
        alpha = 0.9  
        latency = 0.0
        total_pipeline_latency = 0.0
        detected_frames = 0
        

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % skip_stride != 0:
                continue

            tr_start = time()
            
            if annotated is None:
                results = self._predict(frame, stream = True, conf = 0.25)
                for result in results:

                    annotated = result.plot()
                    inference_time = result.speed["inference"]
                    self.logger.info(f"Inference Time: {inference_time:.1f} ms")

            out.write(annotated)

            tr_end = time()
                                            
            # Calculations for latency and fps 
            latency = (tr_end - tr_start) * 1000.0
            fps = 1.0 / (tr_end - tr_start) if (tr_end - tr_start) > 0 else 0.0
            fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * fps)
            fps_smooth = round(fps_smooth)
       
            total_pipeline_latency += latency
            pipeline_fps+= fps_smooth
            detected_frames+=1
            #self.logger.info(f"Frames detected: {detected_frames}")

        cap.release()
        out.release()

         #Calculate Final Global Averages
        if detected_frames > 0:
            avg_latency = total_pipeline_latency / detected_frames
            avg_fps = pipeline_fps/detected_frames
    
        #self.logger.info(f"Average Latency: {avg_latency:.1f} ms")
        #self.logger.info(f"Average FPS: {avg_fps} ")
        
        return output_path
