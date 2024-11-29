"""IMPORT PACKAGES"""

# Import packages
import numpy as np
import cv2
from ultralytics import YOLO
from mask_generation import generate_mask
from gradient_domain_fusion import gradient_domain_fusion_with_transformation
import gradio as gr


# Function to process images and perform blending
def process_images(source_image, target_image, scale, tx, ty, angle):
    # Load the YOLOv11 instance segmentation model
    model = YOLO("yolo11x-seg.pt")  # Adjust the path to your model file

    # Perform inference on source and target images
    src_result = model.predict(source=source_image, save=False)[0]
    tar_result = model.predict(source=target_image, save=False)[0]

    # Generate mask for source image
    mask = generate_mask(source_image, src_result, class_id=0)

    # Perform blending with scaling, translation, and rotation
    result = gradient_domain_fusion_with_transformation(
        source_image, target_image, mask, scale, tx, ty, angle
    )

    # Convert images from BGR to RGB for correct color display
    source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Return the results
    return source_rgb, target_rgb, result_rgb


# Define Gradio interface
def gradio_interface(source_image, target_image, scale, tx, ty, angle):
    # Ensure inputs are valid and process images
    source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

    # Call the process function
    source_rgb, target_rgb, result_rgb = process_images(
        source_image, target_image, scale, tx, ty, angle
    )

    # Return all processed images
    return (
        source_rgb,
        target_rgb,
        result_rgb,
    )


# Define input and output components for Gradio
inputs = [
    gr.Image(type="numpy", label="Upload Source Image"),
    gr.Image(type="numpy", label="Upload Target Image"),
    gr.Slider(0.5, 2.0, value=1.0, label="Scale (0.5-2.0)"),
    gr.Slider(-500, 500, value=0, step=1, label="Translation X (px)"),
    gr.Slider(-500, 500, value=0, step=1, label="Translation Y (px)"),
    gr.Slider(-180, 180, value=0, step=1, label="Rotation Angle (degrees)"),
]

outputs = [
    gr.Image(type="numpy", label="Source Image"),
    gr.Image(type="numpy", label="Target Image"),
    gr.Image(type="numpy", label="Blended Result"),
]

# Create and launch Gradio app
gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=outputs,
    title="No Cameraman Left Behind",
    description="Upload a source and target image, adjust parameters, and see the blended result!",
).launch()
