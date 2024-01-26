import torch
import cv2
from PIL import Image
import numpy as np



CONFIDENCE_THRESHOLD = 0.9
# Load the custom YOLOv5 model
model = torch.hub.load('yolov5', 'custom',
                       path=r'C:\Users\Zwingli\PycharmProjects\helmonzy\helmonzy_Fixed.pt',
                       source='local', force_reload=True)

def main():
    #Input video
    #Helm
    # video_path = r"C:\Users\Zwingli\PycharmProjects\helmonzy\TestKecepatan\Helm\Jarak 3 meter\40\helm_3_40.mp4"

    # #Non-helm
    video_path = r"C:\Users\Zwingli\PycharmProjects\helmonzy\TestKecepatan\Non-helm\Jarak 3 meter\40\nonhelm_3_40.mp4"

    cam = cv2.VideoCapture(video_path)

    # Create a window with a resizable flag
    cv2.namedWindow('YOLO', cv2.WINDOW_NORMAL)

    # Set the window to full screen
    cv2.setWindowProperty('YOLO', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps = 1000
    time_delay = int(1000 / fps)

    model.conf = 0.85

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

        # Extract confidence scores
        confidences = result.pred[0][:, -2].cpu().numpy()
        class_labels = result.pred[0][:, -1].int().cpu().numpy()

        # #print for label motor
        # label = 1  # Set the desired label
        #
        # label_indices = np.where(class_labels == label)[0]
        # label_confidences = confidences[label_indices]
        #
        # if len(label_confidences) > 0:
        #     print(f"Label {label}: Confidence Scores = {label_confidences}")
        #     print("---------------------------------------------------------------")

        # Print for labels 0(helm) and 2(nonhelm)
        for label in [0, 2]:
            label_indices = np.where(class_labels == label)[0]
            label_confidences = confidences[label_indices]
            if len(label_confidences) > 0:
                print(f"Label {label}: Average Confidence = {label_confidences}")
                print("---------------------------------------------------------------")

        if cv2.waitKey(time_delay) & 0xFF == ord('q'):
            break

        cv2.imshow('YOLO', rendered_frame)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
