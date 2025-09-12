from ultralytics import YOLO
import os

def main():
    # Pretrained weight to use
    pretrained_model = "yolov8s.pt"  # automatically downloads if missing
    # Dataset YAML
    data_yaml = r"D:\Certificate-Detection\First Dataset\data.yaml"

    # Save directory
    save_dir = r"D:\Certificate-Detection\First Dataset"
    run_name = "certificate_forgery"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load pretrained YOLOv8 model (will download if not found)
    try:
        model = YOLO(pretrained_model)
        print(f"✅ Loaded {pretrained_model} successfully.")
    except Exception as e:
        print("⚠️ Failed to load pretrained weights:", e)
        print("Trying to redownload the pretrained weights...")
        if os.path.exists(pretrained_model):
            os.remove(pretrained_model)
        model = YOLO(pretrained_model)

    # Train the model
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,                 # GPU 0, or "cpu" if no GPU
        project=save_dir,          # save everything here
        name=run_name
    )

    # Validate after training
    model.val(split="test")

    best_model_path = os.path.join(save_dir, run_name, "weights", "best.pt")
    print(f"\n✅ Training complete! Best model saved at:\n{best_model_path}")

if __name__ == "__main__":
    main()
