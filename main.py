import cv2
import torch
from ultralytics import solutions

points = []
MODEL_TYPE = "yolo11s.pt"
VIDEO_PATH = "./sample2.mp4"
region_points = [
    (150, 350),
    (1100, 350),
    (1300, 600),
    (0, 600)
]
classes_allowed = [2, 3, 5, 7]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Titik len{points}: ({x}, {y})")

        if len(points) == 2:
            print(f"\nGaris selesai: x1={points[0][0]}, y1={points[0][1]}, x2={points[1][0]}, y2={points[1][1]}")

def select_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Running on device: {device.upper()}")
    return device

def vehicle_counting():
    device = select_device()
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        show=True,
        show_conf=False,
        show_labels=False,
        region=region_points,
        model=MODEL_TYPE,
        classes=classes_allowed,
        show_out=True,
        show_in=True,
        line_width=2,
        device=device
    )

    cv2.namedWindow("Vehicle Counting")
    cv2.setMouseCallback("Vehicle Counting", mouse_callback)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = counter(frame)
        print(results.classwise_count)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vehicle_counting()