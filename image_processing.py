from ultralytics import YOLO
from mask_generation import generate_mask
from gradient_domain_fusion import gradient_domain_fusion_with_transformation


def process_images(source_image, target_image, scale, tx, ty, angle):
    # Load YOLO model
    model = YOLO("yolo11x-seg.pt")

    # Perform inference
    src_result = model.predict(source=source_image, save=False)[0]

    # Generate mask
    mask = generate_mask(source_image, src_result, class_id=0)

    # Perform blending
    result = gradient_domain_fusion_with_transformation(
        source_image, target_image, mask, scale, tx, ty, angle
    )

    return result
