#!/usr/bin/env python3
"""
OC-SORT Tracking System with Line Crossing Detection
트래킹 + 라인 크로싱 감지 GUI (리팩토링된 detector 사용)

ESC: Exit
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Crypto.Cipher import AES

from src.lidar.data_reciever import DataReciever
from src.line_crossing.detector import LineCrossingDetector, VirtualLine
from src.tracking.detector_configs import get_config
from src.tracking.engine import ObjectTracker


class TrackingLineCrossingGUI:
    """트래킹 + 라인 크로싱 감지 GUI (리팩토링된 버전)"""

    def __init__(
        self, video_path: str, line_config_path: str = "configs/line_configs.json"
    ):
        self.video_path = video_path
        self.lidar_data_receiver = DataReciever()

        self.line_config_path = line_config_path
        self.cap = None
        self.tracker = None
        self.crossing_detector = LineCrossingDetector()
        self.lines: Dict[str, VirtualLine] = {}

        # Video state
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0

        # Statistics
        self.tracking_time = 0.0
        self.total_detections = 0
        self.total_tracks = 0

        # Display settings
        self.window_name = "OC-SORT Tracking + Line Crossing (Refactored)"

        # 라인 설정 로드
        self.load_line_config()

        self.model_type = 1
        self.init_lidar_detection_model()

    def decrypt_model(self, enc_file_path, key):
        with open(enc_file_path, "rb") as f:
            nonce = f.read(16)
            tag = f.read(16)
            ciphertext = f.read()

        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        decrypted = cipher.decrypt_and_verify(ciphertext, tag)
        return decrypted  # bytes 객체 반환

    def init_lidar_detection_model(self):

        if self.model_type == 1:  # onnx encryption O

            onnx_files = [f for f in os.listdir() if f.endswith(".o.shas")]
            if onnx_files:
                latest_onnx_file = max(onnx_files, key=os.path.getmtime)
                print("ONNX Model:", latest_onnx_file)
            else:
                print("ONNX 파일이 없습니다.")

            key = b"M0d3lS3cur3K3y!!"
            decrypted_model = self.decrypt_model(latest_onnx_file, key)

            # self.session = ort.InferenceSession(
            #     # onnx_path,
            #     # latest_onnx_file,
            #     os.path.join(self.bundle_dir, decrypted_model),
            #     sess_options=ort.SessionOptions(),
            #     # providers=['OpenVINOExecutionProvider']
            #     providers = ['CPUExecutionProvider']
            # )

            self.session = ort.InferenceSession(
                decrypted_model,
                sess_options=ort.SessionOptions(),
                # providers=['OpenVINOExecutionProvider']
                providers=["CUDAExecutionProvider"],
            )

            print(ort.get_available_providers())

    def load_line_config(self):
        """Load line configuration file"""
        try:
            config_path = Path(self.line_config_path)
            if not config_path.exists():
                print(f"Line config file not found: {self.line_config_path}")
                return

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            lines_data = config_data.get("lines", {})

            for line_id, line_data in lines_data.items():
                line = VirtualLine(
                    line_id=line_id,
                    name=line_data.get("name", line_id),
                    start_point=tuple(line_data["start_point"]),
                    end_point=tuple(line_data["end_point"]),
                    color=tuple(line_data.get("color", [0, 255, 0])),
                    thickness=line_data.get("thickness", 3),
                    is_active=line_data.get("is_active", True),
                )
                self.lines[line_id] = line

            print(f"Line config loaded: {len(self.lines)} lines")
            for line_id, line in self.lines.items():
                print(f"  - {line.name}: {line.start_point} -> {line.end_point}")

        except Exception as e:
            print(f"Failed to load line config: {e}")

    def initialize(self, mode: str = "video") -> bool:
        """Initialize video and tracker"""
        try:
            # Open video
            if mode == "video":
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
            elif mode == "lidar":
                print("Initializing Lidar mode...")
                # 라이다 모드에서는 비디오 캡처를 사용하지 않음
                self.fps = 30.0  # 기본 FPS 설정
                self.total_frames = 0  # 라이다는 연속 스트림이므로 0으로 설정
                print("Lidar mode initialized")

            # Initialize tracker with VERY strict settings to minimize ID creation
            # Aggressive settings to prevent ID explosion:
            # - detector_confidence=0.5: Override config to be even more strict
            # - det_thresh=0.4: Only very confident detections
            # - min_hits=5: Require 5 consecutive detections (more strict)
            # - max_age=60: Keep tracks alive even longer
            # - iou_threshold=0.4: More generous association (easier to match)
            config = get_config("high_precision")  # confidence_threshold=0.4
            self.tracker = ObjectTracker(
                det_thresh=0.3,
                max_age=100,
                min_hits=3,  # 3 -> 5 (5번 연속 감지 후 트랙 생성)
                iou_threshold=0.3,
                delta_t=3,
                asso_func="iou",
                inertia=0.2,
                use_byte=True,
                detector_config=config,
                detector_confidence=0.5,  # Override config confidence to 0.5
                enable_image_enhancement=False,
            )

            print(f"Tracker initialized: {config.model_name}")

            # Create GUI window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            return True

        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """컴팩트한 우하단 정보 패널 그리기"""
        annotated_frame = frame.copy()
        img_h, img_w = frame.shape[:2]

        # 통계 정보 가져오기
        stats = self.crossing_detector.get_statistics()

        # 컴팩트한 정보만 표시
        compact_info = [
            f"Frame: {self.current_frame}/{self.total_frames}",
            f"Tracks: {stats['active_tracks']} | Time: {self.tracking_time*1000:.1f}ms",
            f"IN: {stats['total_in']} | OUT: {stats['total_out']}",
        ]

        # 라인별 요약 (한 줄로)
        if stats["line_stats"]:
            line_summary = []
            for line_id, line_stats in stats["line_stats"].items():
                line_name = (
                    self.lines.get(line_id, {}).name
                    if line_id in self.lines
                    else line_id
                )
                # 라인명 축약 (첫 단어만)
                short_name = line_name.split()[0] if line_name else line_id
                line_summary.append(
                    f"{short_name}({line_stats['in']}/{line_stats['out']})"
                )

            compact_info.append(" | ".join(line_summary))

        # 컴팩트한 박스 크기 계산
        box_width = 300  # 기존 450 -> 300으로 축소
        box_height = len(compact_info) * 18 + 12  # 기존 25 -> 18, 패딩 감소

        # 우하단 위치 계산
        box_x = img_w - box_width - 10  # 화면 우측에서 10픽셀 안쪽
        box_y = img_h - box_height - 10  # 화면 하단에서 10픽셀 위

        # 반투명 오버레이 생성
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 0, 0),
            -1,
        )

        # 반투명 블렌딩 (알파 값: 0.8 = 80% 불투명)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

        # 테두리 그리기 (얇게)
        cv2.rectangle(
            annotated_frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (255, 255, 255),
            1,
        )

        # 텍스트 그리기 (작은 폰트)
        for i, text in enumerate(compact_info):
            y = box_y + 15 + i * 18  # 시작 위치와 간격 조정
            color = (0, 255, 255) if "IN:" in text and "OUT:" in text else (0, 255, 0)
            cv2.putText(
                annotated_frame,
                text,
                (box_x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return annotated_frame

    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw virtual lines"""
        for line in self.lines.values():
            if line.is_active:
                cv2.line(
                    frame,
                    line.start_point,
                    line.end_point,
                    line.color,
                    line.thickness * 1,
                )

                # # Display line name
                # text_pos = (line.start_point[0], line.start_point[1] - 10)
                # cv2.putText(
                #     frame,
                #     line.name,
                #     text_pos,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     line.color,
                #     2,
                # )
        return frame

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

        # 라인 크로싱 감지 (리팩토링된 detector 사용)
        crossing_events = self.crossing_detector.detect_crossings(
            tracking_frame.detections, self.lines
        )

        # 비활성 추적 이력 정리
        active_track_ids = [
            d.track_id for d in tracking_frame.detections if d.track_id > 0
        ]
        self.crossing_detector.cleanup_old_tracks(active_track_ids)

        # Draw results
        annotated_frame = frame.copy()

        # 가상 라인 그리기
        annotated_frame = self.draw_lines(annotated_frame)

        # 추적 결과 그리기
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

            # Draw center point
            center_x, center_y = det.center_point
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, color, -1)

            # Draw track path (최근 위치들)
            if track_id > 0 and track_id in self.crossing_detector.track_histories:
                history = self.crossing_detector.track_histories[track_id]
                positions = history.get_recent_positions(5)
                for i in range(1, len(positions)):
                    pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
                    pt2 = (int(positions[i][0]), int(positions[i][1]))
                    cv2.line(annotated_frame, pt1, pt2, color, 2)

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

        # Add compact info panel (우하단)
        annotated_frame = self.draw_info_panel(annotated_frame)

        return annotated_frame

    def detect(self, data: np.ndarray):

        input = np.transpose(data, axes=(1, 0, 2, 3)).astype(dtype=np.float32)
        outputs = self.session.run(["output"], {"input": input})

        return outputs

    def run(self):
        """Main execution loop"""
        if not self.initialize():
            return

        print("Starting tracking + line crossing detection GUI (Refactored)...")
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
            import traceback

            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        # Final statistics output
        stats = self.crossing_detector.get_statistics()
        print("\n" + "=" * 50)
        print("Final Line Crossing Statistics:")
        print(f"Total Entries: {stats['total_in']}")
        print(f"Total Exits: {stats['total_out']}")

        for line_id, line_stats in stats["line_stats"].items():
            line_name = (
                self.lines.get(line_id, {}).name if line_id in self.lines else line_id
            )
            print(f"{line_name}: IN={line_stats['in']}, OUT={line_stats['out']}")

        print("Cleanup completed")

    def run_lidar(self):
        """Main execution loop for Lidar"""
        if not self.initialize("lidar"):
            return

        print("Starting tracking + line crossing detection GUI with Lidar...")
        print("Press ESC to exit")
        print("=" * 50)

        try:
            frame_count = 0
            while True:
                # Read lidar data
                lidar_images = self.lidar_data_receiver.receive_data()

                if lidar_images and self.lidar_data_receiver.sensor_sn:
                    # Get the first sensor's image
                    sensor_id = self.lidar_data_receiver.sensor_sn[0]
                    frame = lidar_images.get(sensor_id)

                    if frame is not None:
                        frame_count += 1
                        self.current_frame = frame_count

                        # Process frame with tracking and line crossing
                        annotated_frame = self.process_frame(frame)

                        # Display processed frame with tracking info
                        cv2.imshow(self.window_name, annotated_frame)

                        # Also show raw lidar data for comparison
                        cv2.imshow("Raw Lidar Data", frame)
                    else:
                        print("Warning: No frame data received from lidar")
                else:
                    print("Warning: No lidar data received")

                # Handle keyboard input (ESC only)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

        except KeyboardInterrupt:
            print("\nUser interrupted")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.cleanup()


def main():
    """Main function"""
    video_path = "data/people.mp4"
    line_config_path = "configs/line_configs.json"

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        print("Available files in data directory:")
        for f in Path("data").glob("*.mp4"):
            print(f"   {f}")
        return

    if not Path(line_config_path).exists():
        print(f"Warning: Line config file not found: {line_config_path}")
        print("Running tracking only without line crossing detection.")

    # Run GUI
    gui = TrackingLineCrossingGUI(video_path, line_config_path)
    gui.run_lidar()


if __name__ == "__main__":
    main()
