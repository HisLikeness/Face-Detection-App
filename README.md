# Face-Detection-App
## Overview
This application uses the Viola-Jones algorithm for face detection, implemented through OpenCV, and is designed as a Streamlit web app. Users can upload images, adjust detection parameters, and save processed images with detected faces. The app also maintains a log of detected faces for review.

## Features
- Detect faces in uploaded images using OpenCV's Haar Cascade Classifier.
- Adjustable detection parameters:
  - **minNeighbors**: Controls the minimum number of rectangles around a candidate face.
  - **scaleFactor**: Determines how much the image size is reduced during detection.
- Customizable rectangle colors for detected faces.
- Save processed images locally.
- Maintain a log of detected faces.

## Requirements
To run this application, ensure you have the following installed:
- Python 3.7+
- Required Python libraries: `opencv-python`, `streamlit`, `numpy`

Install dependencies using pip:
```bash
pip install opencv-python streamlit numpy
```

## How to Use
1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload an image file (JPG, JPEG, or PNG).
3. Adjust the detection parameters as needed:
   - **minNeighbors**: Adjust between 1 and 10 (default is 5).
   - **scaleFactor**: Set between 1.1 and 2.0 (default is 1.1).
4. Choose the color of the rectangles using the color picker.
5. Click the **Detect Faces** button to process the image.
6. View the image with detected faces on the interface.
7. Save the processed image by clicking the **Save Image** button.
8. Review the face detection log (`face_log.txt`).

## File Structure
- `app.py`: The main application file.
- `haarcascade_frontalface_default.xml`: Pre-trained Haar Cascade Classifier for frontal face detection.
- `detected_faces.jpg`: Saved processed images with detected faces.
- `face_log.txt`: Log file containing details of detected faces.

## Code Walkthrough
The app includes the following functionalities:
1. **Load Haar Cascade Classifier**
   ```python
   face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
   ```

2. **Upload and Process Images**
   Users can upload an image file for face detection:
   ```python
   uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
   ```

3. **Adjustable Parameters**
   Allow users to fine-tune detection parameters:
   ```python
   min_neighbors = st.slider("minNeighbors", 1, 10, 5)
   scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.1, step=0.1)
   ```

4. **Face Detection and Visualization**
   Perform face detection and display results:
   ```python
   faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
   for (x, y, w, h) in faces:
       cv2.rectangle(image, (x, y), (x+w, y+h), color_bgr, 2)
   ```

5. **Save Image and Update Log**
   Save the processed image and update the log file:
   ```python
   cv2.imwrite("detected_faces.jpg", image)
   with open("face_log.txt", "a") as log_file:
       log_file.write(f"Detected {len(faces)} faces in the image.\n")
   ```

## Known Limitations
- Only works with uploaded image files (no live camera support).
- Requires pre-trained Haar Cascade XML file.
- Logging is limited to face counts without additional metadata.

## Future Enhancements
- Implement live webcam support for face detection.
- Extend logging to include metadata such as timestamp and image resolution.
- Incorporate deep learning models for enhanced accuracy.
