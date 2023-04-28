from setuptools import setup, find_packages

setup(
    name="face-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.0.0",
        "face_recognition>=1.3.0",
        "numpy>=1.18.0",
        "deepface>=0.0.49",
        "sqlite3>=3.34.0"
    ],
    entry_points={
        "console_scripts": [
            "face_detection= face_detection.main:main",
        ]
    },
    author="Ryan Pleasance",
    description="A face recognition project using OpenCV and DeepFace.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
