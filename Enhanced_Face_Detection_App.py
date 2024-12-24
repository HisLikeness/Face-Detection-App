"""Creating a Streamlit app for advanced face detection using Viola-Jones and deep learning algorithms"""

import cv2
import streamlit as st
import numpy as np
from datetime import datetime
import os
import csv

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('c:\\Users\\HP\\Documents\\STUDY\\DATA SCIENCE AND ANALYSIS - GOMYCODE\\Data Sets\\haarcascade_frontalface_default.xml')

# Create the Streamlit app
st.title("Face Detection App")

# Add instructions to the interface
st.write("This app detects faces in images.")
st.write("To use the app, follow these steps:")
st.write("1. Upload an image by clicking the 'Browse files' button.")
st.write("2. Adjust the parameters as needed:")
st.write("3. Click the 'Detect Faces' button to detect faces in the image.")
st.write("4. To save the image with detected faces, click the 'Save Image' button.")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Add parameters
min_neighbors = st.slider("minNeighbors", 1, 10, 5)
scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.1, step=0.1)

# Add color picker
color = st.color_picker("Choose the color of the rectangles", "#ff0000")

# Convert color to BGR format
color_bgr = (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))

# Add buttons
if st.button("Detect Faces"):
    if uploaded_file is not None:
        try:
            # Read the image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces using the face cascade classifier
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), color_bgr, 2)

            # Display the image
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image with detected faces")

            # Save the image with detected faces
            cv2.imwrite("detected_faces.jpg", image)
            st.success("Image with detected faces saved successfully!")

            # Create a face log
            with open("face_log.txt", "a") as log_file:
                log_file.write(f"Detected {len(faces)} faces in the image.\n")
            st.success("Face log updated successfully!")

            # Add a button to proceed
            if st.button("OK"):
                st.write("Please upload a new image.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    else:
        st.error("Please upload an image file.")
