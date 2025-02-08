import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('C:/Users/Stranger/Desktop/weapon-detection-system/runs/detect/train/weights/best.pt')

def detect_weapons(frame):
    # Perform inference
    results = model.predict(frame, imgsz=640)
    detections = results[0].boxes.data.cpu().numpy()  # Extract detections

    weapons = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        # Ensure class_id matches weapon classes (0 for gun, 1 for knife)
        if int(class_id) in [0, 1] and confidence > 0.5:  # Confidence threshold
            weapons.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))

    return weapons

# Draw bounding boxes for weapons
def draw_weapon_boxes(frame, weapons):
    for (x1, y1, x2, y2, confidence) in weapons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
        label = f"Weapon ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Main function for testing
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Webcam input
    # For testing on video file
    # cap = cv2.VideoCapture('data/weapons/wee.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and draw weapon boxes
        weapons = detect_weapons(frame)
        draw_weapon_boxes(frame, weapons)

        cv2.imshow('Weapon Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
