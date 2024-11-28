"""IMPORT PACKAGES"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from mask_generation import generate_mask
from gradient_domain_fusion import gradient_domain_fusion_with_transformation

# Change the file directory to whichever source or target image you want to use.
sourcePath = "Images/Source/source3a.jpg"
targetPath = "Images/Target/target3b.jpg"

# Read Images
source = cv2.imread(sourcePath)
target = cv2.imread(targetPath)

# Display Images
_, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
ax[0].set_title("Source Image")
ax[1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
ax[1].set_title("Target Image")

# Load the YOLOv11 instance segmentation model
model = YOLO("yolo11x-seg.pt")  # Or other variants like 'yolov11s-seg.pt'

# Perform inference on an image
src_result = model.predict(source=sourcePath, save=True)[0]
tar_result = model.predict(source=targetPath, save=True)[0]

# Transformation parameters
scale = 0.95  # Scale factor
tx = 100  # translation in x direction
ty = 30  # translation in y direction
angle = 0  # rotation degree

# Perform blending with scaling, translation, and rotation
mask = generate_mask(source, src_result, class_id=0)
result = gradient_domain_fusion_with_transformation(
    source, target, mask, scale, tx, ty, angle
)

# Convert images from BGR to RGB for correct color display
source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
mask_rgb = cv2.cvtColor(
    mask, cv2.COLOR_BGR2RGB
)  # Assuming mask is a BGR image (you may need to adjust depending on your implementation)
result_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)


# Create a plot with 4 images in a single row
plt.figure(figsize=(20, 5))

# Display the images in a row
plt.subplot(1, 4, 1)
plt.imshow(source_rgb)
plt.title("Source Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask_rgb)
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(target_rgb)
plt.title("Target Image")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(result_rgb)
plt.title("Result Image")
plt.axis("off")

# Show the plot with all images
plt.tight_layout()
plt.show()

cv2.imwrite(
    "Results/test.jpg",
    result,
)
