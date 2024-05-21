import cv2
import face_recognition
import numpy as np
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
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

def main():
    # Load a known image and get the encoding
    known_image = face_recognition.load_image_file('tanu.jpg')
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            name = "criminal not detected"

            # Check the first match
            if matches[0]:
                name = "Criminal detected"
                winsound.Beep(1000, 500)

            print(name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
