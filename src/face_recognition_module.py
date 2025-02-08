import face_recognition
import os
import cv2
import csv
import numpy as np

# Paths
DATABASE_PATH = "data/face_database/"
BLACKLIST_CSV = "data/blacklisted_faces.csv"

# Load blacklist data
def load_blacklist():
    blacklist = {}
    with open(BLACKLIST_CSV, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            blacklist[row["name"]] = os.path.join(DATABASE_PATH, row["filename"])
    return blacklist

# Encode faces in the database
def encode_faces(blacklist):
    encoded_faces = {}
    for name, filepath in blacklist.items():
        try:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoded_faces[name] = encodings[0]  # Take the first encoding if there are multiple faces in the image
        except Exception as e:
            print(f"Error encoding face for {name}: {e}")
    return encoded_faces

# Recognize faces in a video frame
def recognize_faces(frame, encoded_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    matches = []
    for face_encoding in face_encodings:
        match_found = False  # Flag to check if any match is found
        for name, known_encoding in encoded_faces.items():
            results = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
            if results[0]:
                matches.append(name)
                match_found = True
                break
        
        # If no match found, append "Unknown Person"
        if not match_found:
            matches.append("Unknown Person")
    
    return matches, face_locations

# Check if the recognized person is blacklisted
def is_blacklisted(name, blacklist):
    return name in blacklist

# Function to handle blacklisted persons (Alert or any action you want to take)
def handle_blacklisted_person(name, frame, locations):
    for (top, right, bottom, left) in locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle for blacklisted
        cv2.putText(frame, f"Blacklisted: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Function to handle unknown persons (Different color for unknowns)
def handle_unknown_person(name, frame, locations):
    for (top, right, bottom, left) in locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)  # Yellow rectangle for unknown
        cv2.putText(frame, f"Unknown Person", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# Main function
def main():
    blacklist = load_blacklist()
    encoded_faces = encode_faces(blacklist)

    cap = cv2.VideoCapture(0)  # Use laptop camera
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    # Create a window for the video feed
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        matches, locations = recognize_faces(frame, encoded_faces)

        # Check if the recognized person is blacklisted and handle
        for match, location in zip(matches, locations):
            if is_blacklisted(match, blacklist):
                handle_blacklisted_person(match, frame, [location])
            else:
                if match == "Unknown Person":
                    handle_unknown_person(match, frame, [location])
                else:
                    cv2.putText(frame, match, (location[3], location[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # Check if the user closed the window or pressed 'q' to quit
        if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:  # Window is closed
            print("Camera window closed, exiting...")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
            print("Exiting...")
            break

    # Release camera and destroy all windows properly after the loop ends
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
