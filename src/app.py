import cv2
import logging
import os
from datetime import datetime
from weapon_detection_module import detect_weapons, draw_weapon_boxes
from face_recognition_module import load_blacklist, encode_faces, recognize_faces, is_blacklisted, handle_blacklisted_person, handle_unknown_person
import tkinter as tk
from tkinter import messagebox

# Configure logging
LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def show_popup(message):
    """Show a popup alert."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showwarning("Alert", message)
    root.destroy()

def main():
    # Loading face of blacklisted and encodings
    blacklist = load_blacklist()
    encoded_faces = encode_faces(blacklist)

    # Opening camera ...using laptop camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Camera not accessible.")
        print("Error: Camera not accessible.")
        return

    # Creating a window for the video feed
    cv2.namedWindow("Real Time Detection", cv2.WINDOW_NORMAL)

    # Video writer for recording
    recording = False
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from camera.")
            print("Error: Failed to read frame from camera.")
            break

        # Processing frame for weapon detection
        weapons = detect_weapons(frame)
        draw_weapon_boxes(frame, weapons)
        if weapons:
            logging.info(f"Weapons detected: {weapons}")

        # Processing frame for face recognition
        matches, locations = recognize_faces(frame, encoded_faces)
        for match, location in zip(matches, locations):
            if is_blacklisted(match, blacklist):
                if not recording:
                    # Start recording
                    logging.warning(f"Blacklisted person detected: {match}")
                    print(f"Blacklisted person detected: {match}. Starting recording.")
                    show_popup(f"Blacklisted person detected: {match}")
                    recording = True
                    output_file = os.path.join(
                        "C:/Users/Stranger/Desktop/weapon-detection-system/data/output",
                        f"blacklisted_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    )
                    out = cv2.VideoWriter(
                        output_file,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (frame.shape[1], frame.shape[0]),
                    )
                handle_blacklisted_person(match, frame, [location])
            else:
                if match == "Unknown Person":
                    logging.info("Unknown person detected.")
                    handle_unknown_person(match, frame, [location])
                else:
                    # For known non-blacklisted persons
                    logging.info(f"Known person detected: {match}")
                    cv2.putText(frame, match, (location[3], location[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to the video file if recording
        if recording and out:
            out.write(frame)

        # Display the frame
        cv2.imshow("Real Time Detection", frame)

        # Check for exit conditions
        if cv2.getWindowProperty("Real Time Detection", cv2.WND_PROP_VISIBLE) < 1:
            logging.info("Camera window closed by user.")
            print("Camera window closed, exiting...")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            logging.info("User pressed 'q' to exit.")
            print("Exiting...")
            break

    # Release camera and video writer, and close windows
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    logging.info("Program terminated successfully.")

if __name__ == "__main__":
    main()
