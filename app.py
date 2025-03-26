from HelmetDetector import *
import gradio as gr

# Initialize the detector
path = "SH_yolo11n.pt"
detector = HelmetDetector(path)

#Functions to load image or video
def detect_helmet_image(image):
    """Function for Gradio interface - image input"""
    if image is None:
        return None
    
    result = detector.detect(image)
    return result

def detect_helmet_video(video_path):
    """Function for Gradio interface - video input"""
    if not video_path:
        return None
    
    output_path = detector.process_video(video_path)
    return output_path

# Create Gradio interface
with gr.Blocks(title="Safety Helmet Detection System") as app:
    gr.Markdown("# Helmet Safety Detection System")
    gr.Markdown("Upload an image or video to detect people with or without safety helmets")
    
    with gr.Tabs():
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Input Image")
                    image_button = gr.Button("Detect Safety Helmets")
                with gr.Column():
                    image_output = gr.Image(type="numpy", label="Detection Results")
            
            image_button.click(
                fn=detect_helmet_image,
                inputs=image_input,
                outputs=image_output
            )
        
        with gr.TabItem("Video Detection"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Input Video")
                    video_button = gr.Button("Process Video")
                with gr.Column():
                    video_output = gr.Video(label="Processed Video")
            
            video_button.click(
                fn=detect_helmet_video,
                inputs=video_input,
                outputs=video_output
            )
    
    gr.Markdown("## About")
    gr.Markdown("""
    This application uses YOLOv11 to detect people and determine if they are wearing safety helmets or hard hats.
    - Yellow boxes: Person with helmet
    """)

if __name__ == "__main__":
    # Launch the Gradio app
    app.launch(share=False)  # set share=False for local deployment only
