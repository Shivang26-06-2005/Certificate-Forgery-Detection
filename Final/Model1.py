from ultralytics import YOLO
import os

def main():
    # Define save location
    save_dir = r"D:\Certificate-Detection\Second Dataset\outputs"
    run_name = "certificate_retrain"

    # Make sure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load a pretrained YOLOv8 model
    model = YOLO("yolov8s.pt")   # use yolov8n.pt if GPU memory is low

    # Train the model
    model.train(
        data=r"D:\Certificate-Detection\Second Dataset\data.yaml",  # path to your yaml file
        epochs=80,                  # adjust for your dataset
        imgsz=640,                  # training image size
        batch=4,                    # smaller batch since dataset is small
        workers=2,                  # data loading workers
        device=0,                   # GPU 0, or "cpu" if no GPU
        project=save_dir,           # ✅ custom save directory
        name=run_name               # ✅ subfolder name
    )

    # Validate the model after training
    model.val()

    # Path to best weights
    best_path = os.path.join(save_dir, run_name, "weights", "best.pt")
    print(f"\n✅ Training complete. Best model saved at:\n{best_path}")

if __name__ == "__main__":
    main()
