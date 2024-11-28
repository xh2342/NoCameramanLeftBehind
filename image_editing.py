import cv2
import numpy as np


# Function to crop or pad an image to match the target dimensions
def crop_or_pad_image(image, target_height, target_width):
    src_height, src_width = image.shape[:2]

    # Determine cropping coordinates
    top = max((src_height - target_height) // 2, 0)
    bottom = top + target_height
    left = max((src_width - target_width) // 2, 0)
    right = left + target_width

    # Crop image
    cropped = image[
        max(0, top) : min(src_height, bottom), max(0, left) : min(src_width, right)
    ]

    # Determine padding values
    pad_top = max(-top, 0)
    pad_bottom = max(bottom - src_height, 0)
    pad_left = max(-left, 0)
    pad_right = max(right - src_width, 0)

    # Pad image
    padded = cv2.copyMakeBorder(
        cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    return padded


def crop_or_pad_image(image, target_height, target_width):
    """
    Crop or pad an image to the target dimensions.
    :param image: Input image.
    :param target_height: Desired height.
    :param target_width: Desired width.
    :return: Cropped or padded image.
    """
    src_height, src_width = image.shape[:2]

    # Determine cropping coordinates
    top = max((src_height - target_height) // 2, 0)
    bottom = top + target_height
    left = max((src_width - target_width) // 2, 0)
    right = left + target_width

    # Crop image
    cropped = image[
        max(0, top) : min(src_height, bottom), max(0, left) : min(src_width, right)
    ]

    # Determine padding values
    pad_top = max(-top, 0)
    pad_bottom = max(bottom - src_height, 0)
    pad_left = max(-left, 0)
    pad_right = max(right - src_width, 0)

    # Pad image
    padded = cv2.copyMakeBorder(
        cropped,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded


def scale_image(image, scale, target_height, target_width):
    """
    Scale an image by a given factor and crop/pad to target size.
    :param image: Input image.
    :param scale: Scaling factor (e.g., 1.5 for 150% scaling).
    :param target_height: Desired height.
    :param target_width: Desired width.
    :return: Scaled and cropped/padded image.
    """
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return crop_or_pad_image(scaled_image, target_height, target_width)


def translate_image(image, tx, ty, target_height, target_width):
    """
    Translate an image by (tx, ty) and crop/pad to target size.
    :param image: Input image.
    :param tx: Translation along x-axis.
    :param ty: Translation along y-axis.
    :param target_height: Desired height.
    :param target_width: Desired width.
    :return: Translated and cropped/padded image.
    """
    h, w = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(
        image,
        translation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop_or_pad_image(translated_image, target_height, target_width)


def rotate_image(image, angle, target_height, target_width):
    """
    Rotate an image by a specified angle and crop/pad to target size.
    :param image: Input image.
    :param angle: Rotation angle in degrees.
    :param target_height: Desired height.
    :param target_width: Desired width.
    :return: Rotated and cropped/padded image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)  # Rotate around the center of the image
    rotation_matrix = cv2.getRotationMatrix2D(
        center, angle, 1.0
    )  # 1.0 is the scale factor
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop_or_pad_image(rotated_image, target_height, target_width)
