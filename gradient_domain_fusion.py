from image_editing import *
from skimage.transform import pyramid_gaussian


def gradient_domain_fusion_with_transformation(
    source, target, mask, scale, tx, ty, angle
):
    """
    Perform gradient domain fusion with scaling, translation, and rotation.
    :param source: Source image (cropped and resized).
    :param target: Target image where the source is to be blended.
    :param mask: Binary mask defining the blending region.
    :param scale: Scaling factor for the source image.
    :param tx: Translation along x-axis for the source.
    :param ty: Translation along y-axis for the source.
    :param angle: Rotation angle for the source image.
    :return: Blended image.
    """
    # Get target dimensions
    target_height, target_width = target.shape[:2]

    # Scale, rotate, and translate the source and mask
    transformed_source = translate_image(
        rotate_image(
            scale_image(source, scale, target_height, target_width),
            angle,
            target_height,
            target_width,
        ),
        tx,
        ty,
        target_height,
        target_width,
    )
    transformed_mask = translate_image(
        rotate_image(
            scale_image(mask, scale, target_height, target_width),
            angle,
            target_height,
            target_width,
        ),
        tx,
        ty,
        target_height,
        target_width,
    )

    # Ensure mask is binary and normalized
    transformed_mask = transformed_mask.astype(np.float32) / 255.0

    # Generate Gaussian pyramids for source, target, and mask
    levels = 6  # Number of pyramid levels

    def generate_pyramid(image, levels):
        channels = cv2.split(image)
        pyramids = [
            tuple(pyramid_gaussian(channel, max_layer=levels)) for channel in channels
        ]
        return [
            cv2.merge([pyr[i] for pyr in pyramids]) for i in range(len(pyramids[0]))
        ]

    source_pyramid = generate_pyramid(transformed_source, levels)
    target_pyramid = generate_pyramid(target, levels)
    mask_pyramid = tuple(pyramid_gaussian(transformed_mask, max_layer=levels))

    # Create Laplacian pyramids
    source_laplacian = [
        source_pyramid[i]
        - cv2.resize(source_pyramid[i + 1], source_pyramid[i].shape[:2][::-1])
        for i in range(levels)
    ]
    target_laplacian = [
        target_pyramid[i]
        - cv2.resize(target_pyramid[i + 1], target_pyramid[i].shape[:2][::-1])
        for i in range(levels)
    ]

    # Append the last levels
    source_laplacian.append(source_pyramid[-1])
    target_laplacian.append(target_pyramid[-1])

    # Blend pyramids using the mask
    blended_pyramid = [
        source_laplacian[i] * mask_pyramid[i][:, :, None]
        + target_laplacian[i] * (1 - mask_pyramid[i][:, :, None])
        for i in range(levels + 1)
    ]

    # Reconstruct the blended image
    blended = blended_pyramid[-1]
    for i in range(levels - 1, -1, -1):
        blended = (
            cv2.resize(blended, blended_pyramid[i].shape[:2][::-1]) + blended_pyramid[i]
        )

    # Normalize the result to [0, 255] range and return as uint8
    blended_rescaled = np.clip(
        (blended - blended.min()) / (blended.max() - blended.min()) * 255, 0, 255
    )
    return blended_rescaled.astype(np.uint8)
