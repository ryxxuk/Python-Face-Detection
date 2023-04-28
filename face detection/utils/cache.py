import face_recognition

# Cache settings
cache_size = 100
face_cache = {}

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