import numpy as np
from image_processing import process_images
import cv2


def gradio_interface(source_image, target_image, scale, tx, ty, angle):
    source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    result = process_images(source_image, target_image, scale, tx, ty, angle)
    result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return result_rgb
