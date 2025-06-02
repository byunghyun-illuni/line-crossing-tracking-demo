#!/usr/bin/env python3
"""
OC-SORT Tracking System GUI - Simple Video Player
ESC: Exit
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tracking.detector_configs import get_config
from src.tracking.engine import ObjectTracker


class TrackingGUI:
    """Simple Real-time Tracking GUI"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.tracker = None

        # Video state
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0

        # Statistics
        self.tracking_time = 0.0
        self.total_detections = 0
        self.total_tracks = 0

        # Display settings
        self.window_name = "OC-SORT Tracking System"

    def initialize(self) -> bool:
        """Initialize video and tracker"""
        try:
            # Open video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Error: Cannot open video: {self.video_path}")
                return False

            # Video info
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(
                f"Video: {width}x{height}, {self.fps:.1f}fps, {self.total_frames} frames"
            )

            # Initialize tracker
            config = get_config("crowded_scene")
            self.tracker = ObjectTracker(
                det_thresh=0.1,
                max_age=30,
                min_hits=1,
                iou_threshold=0.3,
                delta_t=3,
                asso_func="iou",
                inertia=0.2,
                use_byte=True,
                detector_config=config,
                enable_image_enhancement=False,
            )

            print(f"Tracker initialized: {config.model_name}")

            # Create GUI window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def draw_simple_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw simple frame info"""
        annotated_frame = frame.copy()

        # Simple info texts
        texts = [
            f"Frame: {self.current_frame}/{self.total_frames}",
            f"Detections: {self.total_detections}",
            f"Tracked: {self.total_tracks}",
            f"Time: {self.tracking_time*1000:.1f}ms",
        ]

        # Draw background box
        box_width = 300
        box_height = len(texts) * 25 + 20
        cv2.rectangle(annotated_frame, (10, 10), (box_width, box_height), (0, 0, 0), -1)
        cv2.rectangle(
            annotated_frame, (10, 10), (box_width, box_height), (255, 255, 255), 2
        )

        # Draw texts
        for i, text in enumerate(texts):
            y = 35 + i * 25
            cv2.putText(
                annotated_frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and tracking"""
        start_time = time.time()

        # Perform tracking
        tracking_frame = self.tracker.track_frame(frame)
        self.tracking_time = time.time() - start_time

        # Update statistics
        self.total_detections = len(tracking_frame.detections)
        self.total_tracks = len(
            [d for d in tracking_frame.detections if d.track_id > 0]
        )

        # Draw results
        annotated_frame = frame.copy()

        for det in tracking_frame.detections:
            x, y, w, h = det.bbox
            track_id = det.track_id

            # Validity check
            img_h, img_w = frame.shape[:2]
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                continue

            # Color and thickness
            if track_id > 0:
                # Unique color based on track ID
                np.random.seed(track_id)
                color = tuple(map(int, np.random.randint(100, 255, 3)))
                thickness = 3
                label = f"ID{track_id}: {det.confidence:.2f}"
            else:
                color = (128, 128, 128)
                thickness = 2
                label = f"?: {det.confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)

            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                annotated_frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Add info overlay
        annotated_frame = self.draw_simple_info(annotated_frame)

        return annotated_frame

    def run(self):
        """Main execution loop"""
        if not self.initialize():
            return

        print("Starting real-time tracking GUI...")
        print("Press ESC to exit")
        print("=" * 50)

        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    continue

                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Process frame
                annotated_frame = self.process_frame(frame)

                # Display
                cv2.imshow(self.window_name, annotated_frame)

                # Handle keyboard input (ESC only)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

        except KeyboardInterrupt:
            print("\nUser interrupted")

        except Exception as e:
            print(f"Error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")


def main():
    """Main function"""
    video_path = "data/people.mp4"

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        print("Available files in data directory:")
        for f in Path("data").glob("*.mp4"):
            print(f"   {f}")
        return

    # Run GUI
    gui = TrackingGUI(video_path)
    gui.run()


if __name__ == "__main__":
    main()
