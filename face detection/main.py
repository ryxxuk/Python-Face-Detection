import sys
import argparse
from detectface import load_known_faces_and_encodings, recognize_faces_from_video, setup_database

def main(args):
    setup_database()
    known_face_encodings, known_face_labels = load_known_faces_and_encodings()
    recognize_faces_from_video(known_face_encodings, known_face_labels)
    
if __name__ == '__main__':
    main(sys.argv[1:])