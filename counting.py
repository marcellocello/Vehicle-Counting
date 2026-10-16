import cv2
import torch
from ultralytics import solutions

class VehicleCounting:
    def __init__(self, video_path="./sample2.mp4", model_type="yolo11s.pt", show_window=False):
        self.video_path = video_path
        self.device = self._select_device()
        self.show_window = show_window
        self.region_points = [
            (150, 350),
            (1100, 350),
            (1300, 600),
            (0, 600)
        ]
        self.classes_allowed = [2, 3, 5, 7]
        self.cap = cv2.VideoCapture(self.video_path)
        assert self.cap.isOpened(), f"Gagal membaca video: {self.video_path}"

        self.counter = solutions.ObjectCounter(
            show=self.show_window,
            show_conf=False,
            show_labels=False,
            region=self.region_points,
            model=model_type,
            classes=self.classes_allowed,
            show_out=True,
            show_in=True,
            line_width=2,
            device=self.device
        )

    def _select_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"Running on device: {device.upper()}")
        return device

    def process_next_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            if not success:
                return {}

        results = self.counter(frame)
        return results.classwise_count

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = VehicleCounting(show_window=True)
    try:
        while True:
            counts = detector.process_next_frame()
            print("Current Counts:", counts)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.release()