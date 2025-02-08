from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

# Path to `data.yaml`
data_path = r"C:/Users/Stranger/Desktop/weapon-detection-system/data/data.yaml"

# Define training parameters
epochs = 30
imgsz = 640

try:
    print("Starting training...")
    model.train(data=data_path, epochs=epochs, imgsz=imgsz, verbose=True)

    print("Training completed successfully!")

    # Export model to ONNX format
    export_path = os.path.join("models", "best.onnx")
    model.export(format="onnx", imgsz=imgsz)
    print(f"Model exported to {export_path}")

except Exception as e:
    print(f"Error: {e}")
