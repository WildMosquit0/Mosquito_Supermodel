from sahi.predict import predict as sahi_predict
from ultralytics import YOLO

# SAHI Prediction with your specific arguments

model = "/home/bohbot/Evyatar/runs/detect/most_update_batch3/weights/best.pt"
source = "/home/bohbot/Evyatar/runs/IMG_7631_frame6576.png"

sahi_predict(
    model_type="ultralytics",  # Use the model type specific to your YOLO version
    model_path= model,
    model_device="cpu",  # Set to 'cuda:0' if using GPU
    model_confidence_threshold=0.1,
    source=source,  # Path to directory containing images
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)


# Load the YOLO model
model = YOLO(model)

# Perform prediction
results = model.predict(
    source=source,
    conf=0.4,  # Confidence threshold for predictions
    save=True,
    imgsz=640  # Save the results
)
