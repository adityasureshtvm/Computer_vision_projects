import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from ultralytics import YOLO
import winsound

# Constants
MODEL_PATH = r"D:\June\CV\YOLO Models\yolov10m.pt"
BAG_CLASSES = [24, 26, 28, 29]  # COCO: backpack, handbag, suitcase, umbrella
PERSON_CLASS = 0
PROXIMITY_THRESHOLD = 250  # pixels
MIN_STATIC_FRAMES = 30  # ~2 seconds at 15 FPS

class UnattendedBagDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.object_history = defaultdict(list)
        self.alert_status = {}  # obj_id: True/False
        self.person_boxes = []

    def get_center(self, box):
        return (int(box[0]), int(box[1]))

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def is_near_person(self, bag_center):
        for p_box in self.person_boxes:
            person_center = self.get_center(p_box)
            if self.calculate_distance(bag_center, person_center) < PROXIMITY_THRESHOLD:
                return True
        return False

    def is_static(self, obj_id):
        history = self.object_history[obj_id]
        if len(history) < MIN_STATIC_FRAMES:
            return False

        movements = [
            self.calculate_distance(history[i][1], history[i - 1][1])
            for i in range(1, len(history))
        ]
        return np.mean(movements) < 10  # pixels/frame

    def trigger_alert(self, frame, box, obj_id):
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 2)
        cv2.putText(frame, "ALERT: Unattended Bag!", (x - w // 2, y - h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Beep sound (repeat each frame while alert is active)
        try:
            winsound.Beep(1000, 300)
        except:
            pass

    def process_frame(self, frame):
        current_time = datetime.now()
        results = self.model.track(frame, persist=True, verbose=False)

        self.person_boxes = []
        current_bags = {}

        if results[0].boxes.id is not None:
            for box, obj_id in zip(results[0].boxes, results[0].boxes.id.cpu().numpy().astype(int)):
                class_id = int(box.cls)
                xywh = box.xywh[0].cpu().numpy()

                if class_id == PERSON_CLASS:
                    self.person_boxes.append(xywh)

                elif class_id in BAG_CLASSES:
                    current_bags[obj_id] = xywh
                    bag_center = self.get_center(xywh)
                    self.object_history[obj_id].append((current_time, bag_center))

                    # Trim history to last 30 seconds
                    self.object_history[obj_id] = [
                        (t, pos) for t, pos in self.object_history[obj_id]
                        if (current_time - t) < timedelta(seconds=30)
                    ]

                    # Draw bag box (green by default)
                    x, y, w, h = map(int, xywh)
                    cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
                    cv2.putText(frame, "Bag", (x - w // 2, y - h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check each tracked bag
        for obj_id, box in current_bags.items():
            bag_center = self.get_center(box)

            if not self.is_near_person(bag_center):
                if len(self.object_history[obj_id]) >= MIN_STATIC_FRAMES:
                    if self.is_static(obj_id):
                        # Activate alert (continuous)
                        self.alert_status[obj_id] = True
            else:
                # Person nearby — clear alert
                self.alert_status.pop(obj_id, None)

        # Draw alert boxes for all currently alerted bags
        for obj_id in self.alert_status:
            if obj_id in current_bags:
                self.trigger_alert(frame, current_bags[obj_id], obj_id)

        return frame

def main():
    cap = None
    try:
        print(f"Loading model from: {MODEL_PATH}")
        detector = UnattendedBagDetector()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video capture")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            processed_frame = detector.process_frame(frame)
            cv2.imshow("Unattended Bag Detector", processed_frame)

            if cv2.waitKey(1) == 27:  # ESC key
                break

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting:")
        print(f"1. Verify the file exists at: {MODEL_PATH}")
        print("2. Check webcam is connected and working")
        print("3. Try running as administrator")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
