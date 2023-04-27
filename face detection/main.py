import sys
import argparse
from detectface import load_known_faces_and_encodings, recognize_faces_from_video

def main(args):
    known_faces_dir = "faces"
    known_face_encodings, known_face_labels = load_known_faces_and_encodings(known_faces_dir)
    recognize_faces_from_video(known_face_encodings, known_face_labels)
    
if __name__ == '__main__':
    main(sys.argv[1:])