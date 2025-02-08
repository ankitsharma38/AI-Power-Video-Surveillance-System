import os
import csv
import shutil

# Constants
DATABASE_PATH = "data/face_database/"
BLACKLIST_CSV = "data/blacklisted_faces.csv"

# Ensure the database directory exists
os.makedirs(DATABASE_PATH, exist_ok=True)

def add_to_blacklist(name, image_file):
    """
    Adds a person's name and their image to the blacklist.

    :param name: Name of the person to add to the blacklist
    :param image_file: The file object of the image being uploaded
    """
    # Generate a unique filename for the image
    image_name = f"{name.replace(' ', '_')}.jpg"
    dest_path = os.path.join(DATABASE_PATH, image_name)
    
    # Save the uploaded image to the database folder
    with open(dest_path, "wb") as dest_file:
        shutil.copyfileobj(image_file, dest_file)
    
    print(f"Image for {name} saved at {dest_path}.")

    # Add the name and image path to the blacklist CSV
    with open(BLACKLIST_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, image_name])
    
    print(f"{name} has been added to the blacklist!")

if __name__ == "__main__":
    name = input("Enter the name of the person: ")
    print("Please upload the image file for the blacklisted person.")
    
    # Simulate file upload
    image_path = input("Enter the path to the image file: ")
    if not os.path.exists(image_path):
        print(f"Error: The image {image_path} does not exist.")
    else:
        with open(image_path, "rb") as image_file:
            add_to_blacklist(name, image_file)
