import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import winsound

def load_image(img_path):
    """Load and preprocess the image for MobileNetV2."""
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_array):
    """Predict the class of an image using MobileNetV2."""
    model = MobileNetV2(weights='imagenet')
    preds = model.predict(img_array)
    return decode_predictions(preds, top=3)[0]

def prepare_training_data(known_image_path):
    """Prepare training data for the recognizer."""
    known_image = cv2.imread(known_image_path)
    gray_known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_known_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    x, y, w, h = faces[0]
    return gray_known_image[y:y+h, x:x+w], 1

def main():
    # Prepare the training data
    known_image_path = './python/tanu.png'
    face, label = prepare_training_data(known_image_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([face], np.array([label]))

    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)

    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_frame = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_frame)
            if confidence < 50:  # You can adjust the confidence threshold
                name = "Criminal detected"
                winsound.Beep(1000, 500)
            else:
                name = "Criminal not detected"

            print(name)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
