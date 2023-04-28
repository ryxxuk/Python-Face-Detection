import sqlite3

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