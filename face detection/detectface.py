import os
import cv2
import face_recognition
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from deepface import DeepFace


def load_known_faces_and_encodings(known_faces_dir):
    known_face_encodings = []
    known_face_labels = []

    for image_name in os.listdir(known_faces_dir):
        image_path = os.path.join(known_faces_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:  # Check if face_encodings is not empty
            face_encoding = face_encodings[0]
            label = os.path.splitext(image_name)[0]

            known_face_encodings.append(face_encoding)
            known_face_labels.append(label)
        else:
            print(f"No faces found in {image_name}. Deleting this image.")
            os.remove(image_path)  # Delete the image

    return known_face_encodings, known_face_labels


def recognize_faces_from_video(known_face_encodings, known_face_labels, frame_downscale_factor=0.5, face_tolerance=0.6, show_processing_time=True, show_landmarks=False):
    video_capture = cv2.VideoCapture(0)
    unknown_faces_dir = "faces"
    frame_upscale_factor = int(1 / frame_downscale_factor)

    def process_frame(frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(
            frame, face_locations)
        return face_locations, face_encodings

    def estimate_face_attributes(face_image):
        demography = DeepFace.analyze(
            face_image, actions=['age', 'gender'], enforce_detection=False)
        print("Demography:", demography)
        age = demography[0]['age']
        gender = demography[0]["dominant_gender"]
        return age, gender

    with ThreadPoolExecutor() as executor:
        while True:
            start_time = time.time()
            ret, frame = video_capture.read()
            frame_resized = cv2.resize(
                frame, (0, 0), fx=frame_downscale_factor, fy=frame_downscale_factor)

            face_locations, face_encodings = executor.submit(
                process_frame, frame_resized).result()

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=face_tolerance)

                label = "Unknown"
                face_image = frame[top*frame_upscale_factor:bottom*frame_upscale_factor,
                                   left*frame_upscale_factor:right*frame_upscale_factor]

                if True in matches:
                    match_index = matches.index(True)
                    label = known_face_labels[match_index]
                else:
                    unknown_face_image = frame_resized[top:bottom, left:right]
                    unknown_face_filename = os.path.join(
                        unknown_faces_dir, f"{uuid.uuid4()}.jpg")
                    cv2.imwrite(unknown_face_filename, unknown_face_image)
                    add_unknown_face_to_known_faces(
                        unknown_face_filename, "New Face", known_face_encodings, known_face_labels)

                if show_landmarks:
                    face_landmarks_list = face_recognition.face_landmarks(
                        frame_resized)
                    draw_landmarks_on_image(
                        frame, face_landmarks_list, frame_upscale_factor)

                age, gender = executor.submit(
                    estimate_face_attributes, face_image).result()
                label = f"{label} - Age: {age} - Gender: {gender}"

                cv2.rectangle(frame, (left*frame_upscale_factor, top*frame_upscale_factor),
                              (right*frame_upscale_factor, bottom*frame_upscale_factor), (0, 0, 255), 2)
                cv2.putText(frame, label, (left*frame_upscale_factor, top*frame_upscale_factor - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if draw_processing_time_on_image:
                processing_time_ms = int((time.time() - start_time) * 1000)
                draw_processing_time_on_image(frame, processing_time_ms)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()


def add_unknown_face_to_known_faces(image_path, label, known_face_encodings, known_face_labels):
    unknown_face = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(unknown_face)

    if face_encodings:  # Check if face_encodings is not empty
        unknown_face_encoding = face_encodings[0]

        known_face_encodings.append(unknown_face_encoding)
        known_face_labels.append(label)


def draw_feature_lines(face, image, frame_upscale_factor, color, thickness):
    for facial_feature, points in face.items():
        scaled_points = (np.array(points) *
                         frame_upscale_factor).round().astype(int)
        is_closed = facial_feature in [
            'left_eyebrow', 'right_eyebrow', 'top_lip', 'bottom_lip']
        cv2.polylines(image, [scaled_points], isClosed=is_closed,
                      color=color, thickness=thickness)


def draw_landmarks_on_image(image, faces, frame_upscale_factor):
    color = (0, 255, 0)
    thickness = 2

    with ThreadPoolExecutor() as executor:
        executor.map(lambda face: draw_feature_lines(
            face, image, frame_upscale_factor, color, thickness), faces)

    return image


def draw_processing_time_on_image(frame, processing_time_ms):
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
