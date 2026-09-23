import os
import sys
import traceback


def heic_to_jpg(input_heic: str, output_jpg: str) -> None:
    """Convert HEIC to JPEG for Ultralytics YOLOE input."""
    try:
        from PIL import Image
        import pillow_heif

        pillow_heif.register_heif_opener()
        image = Image.open(input_heic).convert("RGB")
        image.save(output_jpg, format="JPEG", quality=95)
        print(f"Converted {input_heic} → {output_jpg} ({image.size[0]}x{image.size[1]})")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pillow-heif")
        sys.exit(1)


def main():
    input_heic  = "sorted_data/solar_panel/IMG_5801.HEIC"
    input_jpg   = "sorted_data/solar_panel/IMG_5801.jpg"
    output_dir  = "sorted_data/solar_panel/yoloe_out"

    # Step 1: Convert HEIC → JPEG
    if not os.path.exists(input_jpg):
        heic_to_jpg(input_heic, input_jpg)
    else:
        print(f"{input_jpg} already exists. Delete to force reconversion.")

    # Step 2: Run YOLOE with text prompt "solar panel"
    try:
        from ultralytics import YOLOE

        # yoloe-26l-seg.pt = largest/most accurate variant with segmentation masks
        # On first run, this downloads ~254 MB mobileclip2_b.ts text encoder
        model = YOLOE("yoloe-26l-seg.pt")
        model.set_classes(["solar panel"])

        print("Running YOLOE inference with text prompt: 'solar panel'...")
        results = model.predict(
            source=input_jpg,
            conf=0.15,           # low threshold — panels may not look like COCO objects
            iou=0.45,
            save=True,           # saves annotated image to runs/segment/predict/
            save_dir=output_dir,
            verbose=True,
        )

        # Step 3: Print detections
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        if boxes is None or len(boxes) == 0:
            print("\nNo solar panels detected.")
        else:
            print(f"\nDetected {len(boxes)} solar panel(s):")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                label  = result.names[cls_id]
                conf   = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(
                    f"  [{i+1}] class={label}  conf={conf:.2%}  "
                    f"bbox=({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})"
                )

        print(f"\nAnnotated result saved to: {output_dir}/")

    except ImportError:
        print("ultralytics not installed. Run: pip install -U ultralytics")
        traceback.print_exc()
    except Exception:
        print("Error during inference:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
