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
from screeninfo import get_monitors

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Crypto.Cipher import AES

from src.lidar.data_reciever import DataReciever
from src.line_crossing.detector import LineCrossingDetector, VirtualLine
from src.line_crossing.modes import (
    TrackingPointMode,
    get_mode_description,
    get_tracking_point,
)
from src.tracking.detector_configs import get_config
from src.tracking.engine import ObjectTracker


class TrackingLineCrossingGUI:
    """라이다 트래킹 + 라인 크로싱 감지 GUI"""

    def __init__(
        self,
        line_config_path: str = "configs/line_configs.json",
        tracking_mode: TrackingPointMode = TrackingPointMode.BOTTOM_CENTER,
    ):
        self.lidar_data_receiver = DataReciever()
        self.line_config_path = line_config_path
        self.tracker = None
        self.tracking_mode = tracking_mode
        self.crossing_detector = LineCrossingDetector(tracking_mode=tracking_mode)
        self.lines: Dict[str, VirtualLine] = {}

        # Lidar state
        self.current_frame = 0
        self.fps = 30.0  # 기본 FPS

        # FPS 계산용 변수들
        self.frame_times = []  # 최근 프레임 시간들
        self.last_frame_time = time.time()
        self.fps_update_interval = 1.0  # 1초마다 FPS 업데이트
        self.current_fps = 0.0

        # Statistics
        self.tracking_time = 0.0
        self.total_detections = 0
        self.total_tracks = 0

        # Display settings
        self.window_name = "Lidar Tracking + Line Crossing"

        # 라인 설정 로드
        self.load_line_config()

        self.model_type = 1
        self.init_lidar_detection_model()

        # 추적 모드 정보 출력
        mode_desc = get_mode_description(self.tracking_mode)
        print("라이다 트래킹 시스템 초기화 완료")
        print(f"추적 모드: {mode_desc}")

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

            self.session = ort.InferenceSession(
                decrypted_model,
                sess_options=ort.SessionOptions(),
                # providers=['OpenVINOExecutionProvider']
                providers=["CUDAExecutionProvider"],
            )

            print(ort.get_available_providers())

    def load_line_config(self):
        """Load line configuration file (Multi-Sensor support)"""
        try:
            config_path = Path(self.line_config_path)
            if not config_path.exists():
                print(f"Line config file not found: {self.line_config_path}")
                return

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # 새로운 multi-sensor 구조 처리
            sensors_data = config_data.get("sensors", {})

            if not sensors_data:
                print("No sensor configurations found in config file")
                return

            # 현재 활성 센서 감지 또는 첫 번째 센서 사용
            current_sensor_sn = None
            if self.lidar_data_receiver and self.lidar_data_receiver.sensor_sn:
                current_sensor_sn = str(self.lidar_data_receiver.sensor_sn[0])

            # 현재 센서 설정이 없으면 첫 번째 센서 사용
            if current_sensor_sn not in sensors_data:
                current_sensor_sn = list(sensors_data.keys())[0]
                print(f"Using first available sensor: {current_sensor_sn}")

            sensor_data = sensors_data.get(current_sensor_sn)
            if not sensor_data or not sensor_data.get("line"):
                print(f"No line configuration found for sensor {current_sensor_sn}")
                return

            line_data = sensor_data["line"]

            # 단일 라인 생성 (현재 센서용)
            line_id = f"sensor_{current_sensor_sn}_line"
            line = VirtualLine(
                line_id=line_id,
                name=line_data.get("name", "Detection Line"),
                start_point=tuple(line_data["start_point"]),
                end_point=tuple(line_data["end_point"]),
                color=tuple(line_data.get("color", [0, 255, 0])),
                thickness=line_data.get("thickness", 3),
                is_active=line_data.get("is_active", True),
            )
            self.lines[line_id] = line

            print(f"Line config loaded for sensor {current_sensor_sn}: {line.name}")
            print(f"  - {line.name}: {line.start_point} -> {line.end_point}")

        except Exception as e:
            print(f"Failed to load line config: {e}")

    def initialize(self) -> bool:
        """라이다 트래킹 시스템 초기화"""
        try:
            print("라이다 트래킹 시스템 초기화 중...")

            # 모니터 해상도 정보 가져오기
            monitor = get_monitors()[0]
            screen_width = monitor.width
            screen_height = monitor.height
            print(f"화면 해상도: {screen_width}x{screen_height}")

            # Initialize tracker with optimized settings
            config = get_config("high_precision")  # confidence_threshold=0.4
            self.tracker = ObjectTracker(
                det_thresh=0.3,
                max_age=100,
                min_hits=3,
                iou_threshold=0.3,
                delta_t=3,
                asso_func="iou",
                inertia=0.2,
                use_byte=True,
                detector_config=config,
                detector_confidence=0.5,
                enable_image_enhancement=False,
            )

            print(f"트래커 초기화 완료: {config.model_name}")

            # Create GUI window with screen resolution
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, screen_width, screen_height)

            return True

        except Exception as e:
            print(f"초기화 실패: {e}")
            return False

    def draw_info_panel(self, frame: np.ndarray) -> np.ndarray:
        """우하단 정보 표시 (간결하게)"""
        annotated_frame = frame.copy()
        img_h, img_w = frame.shape[:2]

        # 통계 정보 가져오기
        stats = self.crossing_detector.get_statistics()

        # 간결한 정보로 압축
        mode_short = self.tracking_mode.value.upper()[:3]  # 모드 약어 (예: BOT, CEN)
        info_lines = [
            f"FPS:{self.current_fps:.1f} T:{stats['active_tracks']} M:{mode_short}",
            f"IN:{stats['total_in']} OUT:{stats['total_out']}",
        ]

        # 우하단에 한 줄씩 표시
        for i, text in enumerate(info_lines):
            # 텍스트 크기 계산
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]

            x = img_w - text_size[0] - 8
            y = img_h - (len(info_lines) - i) * 15 - 8

            # 색상 구분
            color = (0, 255, 255) if "IN:" in text else (0, 255, 0)

            cv2.putText(
                annotated_frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,  # 0.5 -> 0.35로 더 작게
                color,
                1,  # 두께 2 -> 1로 축소
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

        # 좌상단에 센서 SN 표시
        if self.lidar_data_receiver.sensor_sn:
            current_sensor_sn = str(self.lidar_data_receiver.sensor_sn[0])
            sensor_text = f"Sensor SN: {current_sensor_sn}"

            # 센서 SN 텍스트 (박스 없이)
            cv2.putText(
                annotated_frame,
                sensor_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # 0.6 -> 0.4로 축소
                (0, 255, 255),  # 노란색
                1,  # 두께도 2 -> 1로 축소
            )

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

            # Draw tracking point (모드에 따라 다른 위치 - 크고 명확하게)
            tracking_point = get_tracking_point(det, self.tracking_mode)
            track_x, track_y = tracking_point

            # 디버깅: 첫 번째 트랙만 좌표 출력 (너무 많은 로그 방지)
            if track_id == 1 and self.current_frame % 30 == 0:  # 1초마다 한 번
                center_x, center_y = det.center_point
                print(
                    f"ID{track_id}: 중심({center_x:.0f},{center_y:.0f}) -> 추적({track_x:.0f},{track_y:.0f}) [모드:{self.tracking_mode.value}]"
                )

            # 추적 모드별 다른 모양으로 표시 (더 크고 명확하게)
            if self.tracking_mode == TrackingPointMode.BOTTOM_CENTER:
                # 발 추적: 큰 사각형으로 표시
                cv2.rectangle(
                    annotated_frame,
                    (int(track_x) - 8, int(track_y) - 8),
                    (int(track_x) + 8, int(track_y) + 8),
                    color,
                    -1,
                )
                cv2.rectangle(
                    annotated_frame,
                    (int(track_x) - 8, int(track_y) - 8),
                    (int(track_x) + 8, int(track_y) + 8),
                    (255, 255, 255),
                    2,
                )  # 흰 테두리
                # 추가로 발 표시를 위한 작은 선
                cv2.line(
                    annotated_frame,
                    (int(track_x) - 5, int(track_y)),
                    (int(track_x) + 5, int(track_y)),
                    (255, 255, 255),
                    2,
                )
            else:
                # 다른 모드: 큰 원으로 표시
                cv2.circle(annotated_frame, (int(track_x), int(track_y)), 9, color, -1)
                cv2.circle(
                    annotated_frame, (int(track_x), int(track_y)), 9, (255, 255, 255), 2
                )

            # Draw center point (연한 색상으로 참고용 - 더 작게)
            center_x, center_y = det.center_point
            cv2.circle(
                annotated_frame, (int(center_x), int(center_y)), 2, (100, 100, 100), 1
            )

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
        """라이다 트래킹 메인 실행 루프"""
        if not self.initialize():
            return

        print("라이다 트래킹 + 라인 크로싱 감지 시작...")
        print("ESC 키를 누르면 종료됩니다")
        print("=" * 50)

        try:
            frame_count = 0
            while True:
                # 라이다 데이터 수신
                lidar_images = self.lidar_data_receiver.receive_data()

                if lidar_images and self.lidar_data_receiver.sensor_sn:
                    # 첫 번째 센서의 이미지 가져오기
                    sensor_id = self.lidar_data_receiver.sensor_sn[0]
                    frame = lidar_images.get(sensor_id)

                    if frame is not None:
                        frame_count += 1
                        self.current_frame = frame_count

                        # FPS 업데이트
                        self.update_fps()

                        # 프레임 처리 (트래킹 + 라인 크로싱 감지)
                        annotated_frame = self.process_frame(frame)

                        # 처리된 프레임 표시
                        cv2.imshow(self.window_name, annotated_frame)

                    else:
                        print("Warning: 라이다 프레임 데이터가 없습니다")
                else:
                    print("Warning: 라이다 데이터를 받지 못했습니다")

                # 키보드 입력 처리 (ESC만)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

        except KeyboardInterrupt:
            print("\n사용자 중단")

        except Exception as e:
            print(f"오류: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """정리 작업"""
        cv2.destroyAllWindows()

        # 최종 통계 출력
        stats = self.crossing_detector.get_statistics()
        print("\n" + "=" * 50)
        print("최종 라인 크로싱 통계:")
        print(f"총 진입: {stats['total_in']}")
        print(f"총 퇴장: {stats['total_out']}")

        for line_id, line_stats in stats["line_stats"].items():
            line_name = self.lines.get(
                line_id,
                VirtualLine(line_id="", name="", start_point=(0, 0), end_point=(0, 0)),
            ).name
            print(f"{line_name}: IN={line_stats['in']}, OUT={line_stats['out']}")

        print("정리 완료")

    def update_fps(self):
        """FPS 계산 및 업데이트"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # 최근 1초간의 프레임만 유지
        self.frame_times = [t for t in self.frame_times if current_time - t <= 1.0]

        # FPS 계산 (최근 1초간 프레임 수)
        if len(self.frame_times) > 1:
            self.current_fps = len(self.frame_times) - 1  # 마지막 프레임 제외
        else:
            self.current_fps = 0.0


def main():
    """메인 함수"""
    line_config_path = "configs/line_configs.json"

    if not Path(line_config_path).exists():
        print(f"Warning: 라인 설정 파일을 찾을 수 없습니다: {line_config_path}")
        print("라인 크로싱 감지 없이 트래킹만 실행됩니다.")

    # 추적 모드 설정 (필요에 따라 변경 가능)
    tracking_mode = TrackingPointMode.BOTTOM_CENTER  # 발 추적용
    # tracking_mode = TrackingPointMode.CENTER      # 기존 중심점
    # tracking_mode = TrackingPointMode.TOP_CENTER  # 머리 추적용

    print(f"선택된 추적 모드: {get_mode_description(tracking_mode)}")

    # 라이다 트래킹 GUI 실행
    gui = TrackingLineCrossingGUI(line_config_path, tracking_mode)
    gui.run()


if __name__ == "__main__":
    main()
