import numpy as np
import cv2


def generate_mask(img, result, class_id=0):
    """
    Extracts the mask for a specific class from the segmentation results.

    :param img: The input image (in BGR format).
    :param result: The result of the segmentation, containing masks, boxes, and classes.
    :param class_id: The class ID to identify, default is 0.
    :return: The binary mask for the specified class (resized to match the image size), or None if not found.
    """
    # Access the segmentation masks and classes
    masks = result.masks.data  # Contains masks for each detected object
    classes = result.boxes.cls  # Class labels

    # Iterate through the masks and classes
    for mask, cls in zip(masks, classes):
        if cls == class_id:  # Filter by the specified class ID
            # Convert the mask to a binary mask (numpy array)
            mask_np = mask.cpu().numpy()
            mask_np = np.squeeze(mask_np)  # Remove any singleton dimensions

            # Resize the mask to match the image size
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

            # Threshold the mask to make it binary (0 or 255)
            _, mask_binary = cv2.threshold(mask_resized, 0.5, 255, cv2.THRESH_BINARY)

            # Convert to uint8 for compatibility with image processing
            mask_binary = mask_binary.astype(np.uint8)

            return mask_binary

    # Return None if no mask for the specified class is found
    print("no mask for the specified class is found")
    return None
