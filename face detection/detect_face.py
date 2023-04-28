import cv2
import face_recognition
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from deepface import DeepFace

from utils.database import add_face_to_database
from utils.cache import update_cache, is_face_in_cache
from utils.drawing import draw_landmarks_on_image, draw_processing_time_on_image

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