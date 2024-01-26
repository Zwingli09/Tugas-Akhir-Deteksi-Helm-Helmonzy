import os

import torch
from datetime import datetime, timedelta
import cv2
from PIL import Image
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
import io
import pyrebase
from config import config

# Firebase cred
cred = credentials.Certificate(r".\serviceAccount.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'https://helmonzy-default-rtdb.firebaseio.com/images'})

# Load the custom YOLOv5 model
model = torch.hub.load('yolov5', 'custom',
                       path=r'C:\Users\Zwingli\PycharmProjects\helmonzy\helmonzy_Fixed.pt',
                       source='local', force_reload=True)

CONFIDENCE_THRESHOLD = 0.9
cooldown_duration = timedelta(seconds=4)
last_capture_time = datetime.min


def check_object(class_labels, pred, rendered_frame, cm1_frame,cm2_frame, nobox, current_time):
    for index, label in enumerate(class_labels.numpy()):
        if label == 1:
            # Check if the label is 1
            x1, y1, width, height = pred[index, :4].cpu().numpy()

            # Calculate x_min, y_min, x_max, y_max
            x_center = int(x1 + width) // 2
            y_center = int(y1 + height) // 2

            # Mencari centre coordinates
            center_coordinates = (int(x_center), int(y_center))
            radius = 5  # You can adjust the radius based on your preference

            # visual kordinat tengah
            cv2.circle(rendered_frame, center_coordinates, radius, (0, 255, 0), -1)

            if cm2_frame < x_center < cm1_frame:
                # Check cooldown waktu capture
                if (current_time - last_capture_time) < cooldown_duration:
                    print(current_time)
                    print(last_capture_time)
                    print("Cooldown period. Skipping image capture.")
                    return

                #Mengirim gambar untuk diproses pelanggarannya
                process_image(nobox)


def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

    fps = 1000
    time_delay = int(1000 / fps)

    model.conf = 0.9

    while True:
        ret, frame = cam.read()

        # Check if the frame is successfully read
        if not ret:
            print("Error reading frame. Exiting...")
            break

        nobox = frame
        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        result = model(frame, size=1200)
        rendered_frame = np.squeeze(result.render())



        if cv2.waitKey(time_delay) & 0xFF == ord('q'):
            break

        # Center line position
        frame_height, frame_width, _ = frame.shape
        center_frame_width = frame_width // 2

        # Area deteksi 1 (pemilahan motor)
        cm1_frame = center_frame_width + 100
        cm2_frame = center_frame_width - 100


        pred = result.pred[0]

        # Extract class labels
        class_labels = pred[:, -1].int().cpu()
        area_rectangle = [(cm1_frame, 0), (cm2_frame, 0), (cm2_frame, 1200), (cm1_frame, 1200)]


        # go to check object function
        check_object(class_labels, pred, rendered_frame, cm1_frame,cm2_frame, nobox, datetime.now())

        #Visual daerah deteksi
        cv2.polylines(rendered_frame, [np.array(area_rectangle, np.int32)], True, (0, 0, 0), 2)
        cv2.imshow('YOLO', rendered_frame)

    cam.release()
    cv2.destroyAllWindows()


def process_image(image):
    current_time = datetime.now()
    date = current_time.strftime("%d-%m-%Y")
    time = current_time.strftime("%H:%M:%S")

    # Timestamp
    timestamp = {"timestamp": {".sv": "timestamp"}}

    #Directory
    save_directory = r'C:\Users\Zwingli\PycharmProjects\helmonzy\Screenshot'
    image_path = os.path.join(save_directory, f"image_{current_time.strftime('%Y%m%d%H%M%S')}.jpeg")


    # Save image in-memory
    _, image_bytes = cv2.imencode('.jpg', image)
    image_stream = io.BytesIO(image_bytes.tobytes())

    # Load the gambar in-memory
    saved_image = Image.open(image_stream)
    saved_image = np.array(saved_image)

    # Perform YOLO inference on the saved image
    saved_result = model(saved_image)

    # Access the prediction tensor for the saved image
    saved_pred = saved_result.pred[0]


    # Center line position
    frame_height, frame_width, _ = image.shape
    center_frame_height = frame_height // 2
    center_frame_width = frame_width // 2

    # Define the rectangle detection area in the middle of the frame
    cnh1_frame = center_frame_width + 250
    cnh2_frame = center_frame_width - 250

    # Extract class labels for the saved image
    saved_class_labels = saved_pred[:, -1].int().cpu()

    # Print class labels
    print("Image label detected:", saved_class_labels.numpy())

    #Deteksi label nonhelm pada gambar
    if 2 in saved_class_labels.numpy():
        # Find the index of the first occurrence of label 2
        index = np.where(saved_class_labels.numpy() == 2)[0][0]

        x1, y1, width, height = saved_pred[index, :4].cpu().numpy()

        # Calculate x_min, y_min, x_max, y_max
        x_center2 = int(x1 + width) // 2
        y_center2 = int(y1 + height) // 2

        center_coordinates = (int(x_center2), int(y_center2))
        radius = 5  # You can adjust the radius based on your preference

        cv2.circle(image, center_coordinates, radius, (0, 255, 0), -1)

        if cnh2_frame < x_center2 < cnh1_frame:
            saved_imageRGB = cv2.cvtColor(np.array(saved_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, saved_imageRGB)
            print("Gambar tersimpan di lokal, bersiap untuk mengirimkan ke database....")

            # Auto upload gambar ke database firebase
            firebase = pyrebase.initialize_app(config)
            storage = firebase.storage()

            blob_path = f"{current_time.strftime('%Y%m%d%H%M%S')}.jpeg"

            # Upload gambar ke Firebase Storage
            blob = storage.child(blob_path)
            blob.put(image_path, content_type='image/jpeg')

            # Upload data reference gambar
            url = storage.child(blob_path).get_url(None)

            image_data = {
                "image_url": url,
                "date": date,
                "time": time,
                **timestamp
            }
            print(f"Processing image_data with {image_data} at {current_time}")

            firebase_db = firebase.database()
            firebase_db.child().push(image_data)
            print("Data berhasil disimpan di Firebase Realtime Database")
            # update cooldown sistem
            global last_capture_time
            last_capture_time = current_time

if __name__ == "__main__":
    main()
