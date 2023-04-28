import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
