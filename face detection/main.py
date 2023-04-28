import sys
import argparse
from detect_face import recognize_faces_from_video
from utils.database import setup_database, load_known_faces_and_encodings

def main(args):
    setup_database()
    known_face_encodings, known_face_labels = load_known_faces_and_encodings()
    recognize_faces_from_video(known_face_encodings, known_face_labels)
    
if __name__ == '__main__':
    main(sys.argv[1:])