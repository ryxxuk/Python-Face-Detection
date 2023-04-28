import os
import cv2
import face_recognition
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from deepface import DeepFace
import sqlite3

# Cache settings
cache_size = 100
face_cache = {}

# Database settings
db_path = "faces.db"


def setup_database():
    """Create a database table for storing known faces if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS known_faces (
                      id INTEGER PRIMARY KEY,
                      label TEXT,
                      encoding BLOB,
                      age INTEGER,
                      gender TEXT
                      )""")
    conn.commit()
    conn.close()


def load_known_faces_and_encodings():
    """Load known face encodings and their associated data from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM known_faces")
    rows = cursor.fetchall()

    known_face_encodings = []
    known_face_data = []

    for row in rows:
        face_encoding = np.frombuffer(row[2], dtype=np.float64)
        known_face_encodings.append(face_encoding)
        known_face_data.append((row[1], row[3], row[4]))

    conn.close()

    return known_face_encodings, known_face_data


def add_face_to_database(label, face_encoding, age, gender):
    """Add a new face to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO known_faces (label, encoding, age, gender) VALUES (?, ?, ?, ?)",
                   (label, face_encoding.tobytes(), age, gender))
    conn.commit()
    conn.close()

def update_cache(label, face_encoding, age, gender):
    """Update the face cache with a new face."""
    if len(face_cache) >= cache_size:
        face_cache.pop(next(iter(face_cache)))  # Remove the oldest entry
    face_cache[label] = {"encoding": face_encoding,
                         "age": age, "gender": gender}

def is_face_in_cache(face_encoding):
    """Check if a face is present in the cache."""
    for label, face_data in face_cache.items():
        if face_recognition.compare_faces([face_data["encoding"]], face_encoding, tolerance=0.6):
            return label, face_data["age"], face_data["gender"]
    return None, None, None

def process_frame(frame):
    """Process the given frame to find face locations and encodings."""
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    return face_locations, face_encodings


def estimate_face_attributes(face_image):
    """Estimate age and gender of a face using the DeepFace library."""
    demography = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)
    print("Demography:", demography)
    age = demography[0]['age']
    gender = demography[0]["dominant_gender"]
    return age, gender

def draw_feature_lines(face, image, frame_upscale_factor, color, thickness):
    """Draw the feature lines of the face on the image."""
    for facial_feature, points in face.items():
        scaled_points = (np.array(points) * frame_upscale_factor).round().astype(int)
        is_closed = facial_feature in ['left_eyebrow', 'right_eyebrow', 'top_lip', 'bottom_lip']
        
        cv2.polylines(image, [scaled_points], isClosed=is_closed,
                      color=color, thickness=thickness)


def draw_landmarks_on_image(image, faces, frame_upscale_factor):
    """Draw face landmarks on the given image."""
    color = (0, 255, 0)
    thickness = 2

    with ThreadPoolExecutor() as executor:
        executor.map(lambda face: draw_feature_lines(
            face, image, frame_upscale_factor, color, thickness), faces)

    return image


def draw_processing_time_on_image(frame, processing_time_ms):
    """Draw the processing time on the image."""
    processing_time_text = f"Processing time: {processing_time_ms} ms"

    # Get frame width and height
    frame_height, frame_width, _ = frame.shape

    # Calculate the text size and position
    text_size, _ = cv2.getTextSize(
        processing_time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_width, text_height = text_size
    text_x = frame_width - text_width - 10
    text_y = text_height + 10

    # Display the processing time on the top right corner of the video feed
    cv2.putText(frame, processing_time_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def recognize_faces_from_video(known_face_encodings, known_face_data, frame_downscale_factor=0.5, face_tolerance=0.6, show_processing_time=True, show_landmarks=False):
    """Recognize faces from a video feed, estimate their age and gender, and update the database."""
    video_capture = cv2.VideoCapture(0)
    frame_upscale_factor = int(1 / frame_downscale_factor)

    with ThreadPoolExecutor() as executor:
        while True:
            start_time = time.time()
            ret, frame = video_capture.read()
            frame_resized = cv2.resize(
                frame, (0, 0), fx=frame_downscale_factor, fy=frame_downscale_factor)

            face_locations, face_encodings = executor.submit(
                process_frame, frame_resized).result()

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                label, age, gender = is_face_in_cache(face_encoding)
                if label is None:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding, tolerance=face_tolerance)

                    if True in matches:
                        match_index = matches.index(True)
                        label, age, gender = known_face_data[match_index]
                        update_cache(label, face_encoding, age, gender)
                    else:
                        face_image = frame[top * frame_upscale_factor:bottom * frame_upscale_factor,
                                        left * frame_upscale_factor:right * frame_upscale_factor]
                        age, gender = executor.submit(
                            estimate_face_attributes, face_image).result()
                        label = f"New Face {uuid.uuid4()}"
                        add_face_to_database(label, face_encoding, age, gender)

                if show_landmarks:
                    face_landmarks_list = face_recognition.face_landmarks(
                        frame_resized)
                    draw_landmarks_on_image(
                        frame, face_landmarks_list, frame_upscale_factor)

                cv2.rectangle(frame, (left*frame_upscale_factor, top*frame_upscale_factor),
                              (right*frame_upscale_factor, bottom*frame_upscale_factor), (0, 0, 255), 2)
                cv2.putText(frame, label, (left*frame_upscale_factor, top*frame_upscale_factor - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if show_processing_time:
                processing_time_ms = int((time.time() - start_time) * 1000)
                draw_processing_time_on_image(frame, processing_time_ms)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()


def add_unknown_face_to_known_faces(image_path, label, known_face_encodings, known_face_labels):
    """Add an unknown face to the list of known faces."""
    unknown_face = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(unknown_face)

    if face_encodings:  # Check if face_encodings is not empty
        unknown_face_encoding = face_encodings[0]

        known_face_encodings.append(unknown_face_encoding)
        known_face_labels.append(label)