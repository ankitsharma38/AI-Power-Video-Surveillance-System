**🚀 Project Title & Tagline**
Face Recognition and Weapon Detection System with Machine Learning
---------------------------

**Tagline:** "A Python-based system that detects faces and weapons using machine learning models, and provides a user-friendly interface for recording and reviewing footage."

**📖 Description**
The Face Recognition and Weapon Detection System is a Python-based project that utilizes machine learning models to detect faces and weapons in video footage. The system is designed to be used in the field of surveillance and law enforcement, where it can be used to identify individuals and detect potential threats.

The system consists of two main components: face recognition and weapon detection. The face recognition component uses a pre-trained model to identify individuals in the footage, while the weapon detection component uses a YOLOv8 model to detect weapons. The system also includes a user-friendly interface for recording and reviewing footage, as well as a feature for updating the blacklist of faces.

**✨ Features**

1. **Face Recognition**: The system uses a pre-trained face recognition model to identify individuals in the footage.
2. **Weapon Detection**: The system uses a YOLOv8 model to detect weapons in the footage.
3. **Blacklist Feature**: The system allows for the creation of a blacklist of faces, which can be used to identify individuals who are not allowed on the premises.
4. **User-Friendly Interface**: The system includes a user-friendly interface for recording and reviewing footage.
5. **Multi-Camera Support**: The system can be used with multiple cameras to detect faces and weapons in multiple locations.
6. **Real-Time Detection**: The system can detect faces and weapons in real-time.
7. **Storage**: The system includes a storage feature to save the detected faces and weapons for future reference.
8. **Data Analysis**: The system includes data analysis feature to analyze the detected faces and weapons.

**🧰 Tech Stack Table**

| **Frontend** | **Backend** | **Tools** |
| --- | --- | --- |
| Python | Python | OpenCV, YOLOv8, CSV, NumPy, Tkinter |

**📁 Project Structure**

* `face_recognition_module.py`: Contains the face recognition model and its related functions.
* `app.py`: Contains the main application logic and its related functions.
* `main_recorded.py`: Contains the main recorded function and its related functions.
* `update_blacklist.py`: Contains the update blacklist function and its related functions.
* `weapon_detection_module.py`: Contains the weapon detection model and its related functions.
* `train_model.py`: Contains the training function for the YOLOv8 model.
* `__init__.py`: Contains the initialization function for the project.
* `types.py`: Contains the type definitions for the project.
* `client.py`: Contains the client function and its related functions.
* `retry_options.py`: Contains the retry options function and its related functions.
* `_multidict_py.py`: Contains the multidict implementation.
* `_compat.py`: Contains the compatibility functions.
* `_multidict_base.py`: Contains the multidict base implementation.
* `_abc.py`: Contains the abstract base class implementation.

**⚙️ How to Run**

1. **Setup**: Install the required dependencies using pip: `pip install opencv-python numpy yolo-py`
2. **Environment**: Set the environment variable `DATABASE_PATH` to the path where the face database is stored.
3. **Build**: Build the project by running the `main_recorded.py` file.
4. **Deploy**: Deploy the project by copying the `app.py` file to the deployment directory.

**🧪 Testing Instructions**

1. **Face Recognition**: Test the face recognition feature by recording a video and then using the `load_blacklist` function to load the blacklist of faces.
2. **Weapon Detection**: Test the weapon detection feature by recording a video and then using the `detect_weapons` function to detect weapons.
3. **Blacklist Feature**: Test the blacklist feature by adding a face to the blacklist and then using the `is_blacklisted` function to check if the face is blacklisted.

**📸 Screenshots**


**👤 Author**

* Ankit Sharma


This README file provides a comprehensive overview of the Face Recognition and Weapon Detection System project. It includes information on the project's features, tech stack, project structure, and how to run the project. It also includes testing instructions and screenshots of the user interface.
