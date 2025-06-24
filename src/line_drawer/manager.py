#!/usr/bin/env python3
"""
라인 그리기 도구 (Multi-Sensor 지원)

라이다 웹캠 화면에서 마우스로 가상 라인을 그리고 설정합니다.
sensor_sn별로 개별 라인 설정을 관리할 수 있습니다.

사용법:
- 마우스 왼쪽 클릭: 점 설정
- 'S' 키: 라인 저장
- 'R' 키: 초기화
- 'ESC' 키: 종료
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.lidar.data_reciever import DataReciever


class LineDrawer:
    """라이다 웹캠에서 가상 라인 그리기 도구 (Multi-Sensor 지원)"""

    def __init__(self, config_path: str = "configs/line_configs.json"):
        self.config_path = config_path
        self.lidar_receiver = DataReciever()

        # 센서 정보
        self.current_sensor_sn = None
        self.available_sensors = []

        # 라인 그리기 상태
        self.points: List[Tuple[int, int]] = []  # 클릭한 점들
        self.drawing_mode = True
        self.line_completed = False

        # 라인 설정
        self.line_name = "Detection Line"
        self.line_color = (0, 0, 255)  # BGR: 빨간색
        self.line_thickness = 4

        # 화면 설정
        self.window_name = "Line Drawer - Lidar View"

        # 마우스 이벤트 상태
        self.mouse_pos = (0, 0)

        # 기존 설정 로드
        self.load_existing_config()

        print("라인 그리기 도구 초기화 완료 (Multi-Sensor)")
        print("=" * 50)
        print("사용법:")
        print("  1) 마우스 왼쪽 클릭으로 시작점 설정")
        print("  2) 마우스 왼쪽 클릭으로 끝점 설정")
        print("  3) 'S' 키로 라인 저장")
        print("")
        print("키 조작:")
        print("  S = 라인 저장     R = 라인 초기화")
        print("  ESC = 종료")
        print("=" * 50)

    def load_existing_config(self):
        """기존 설정 로드"""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                print("기존 설정 파일이 없습니다. 새로 생성합니다.")
                return

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # 센서 정보 로드
            sensors_data = config_data.get("sensors", {})
            self.available_sensors = list(sensors_data.keys())

            print(f"기존 설정에서 {len(self.available_sensors)}개 센서 정보 로드됨")
            for sensor_sn in self.available_sensors:
                print(f"  - Sensor {sensor_sn}")

        except Exception as e:
            print(f"설정 로드 실패: {e}")

    def detect_current_sensor(self) -> str:
        """현재 활성 센서 감지"""
        if self.lidar_receiver.sensor_sn:
            return str(self.lidar_receiver.sensor_sn[0])
        return None

    def update_current_sensor(self):
        """현재 센서 정보 업데이트"""
        detected_sensor = self.detect_current_sensor()
        if detected_sensor and detected_sensor != self.current_sensor_sn:
            self.current_sensor_sn = detected_sensor
            print(f"센서 변경 감지: {self.current_sensor_sn}")
            self.load_sensor_line()

    def load_sensor_line(self):
        """현재 센서의 기존 라인 로드"""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                return

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            sensor_data = config_data.get("sensors", {}).get(self.current_sensor_sn)
            if sensor_data and sensor_data.get("line"):
                line_data = sensor_data["line"]
                # 기존 라인 정보를 점들로 로드
                self.points = [
                    tuple(line_data["start_point"]),
                    tuple(line_data["end_point"]),
                ]
                self.line_name = line_data.get("name", "Detection Line")
                self.line_color = tuple(line_data.get("color", [0, 0, 255]))
                self.line_thickness = line_data.get("thickness", 4)
                self.line_completed = True
                print(f"센서 {self.current_sensor_sn}의 기존 라인 로드됨")
            else:
                print(f"센서 {self.current_sensor_sn}의 라인 설정이 없습니다")

        except Exception as e:
            print(f"센서 라인 로드 실패: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 함수"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_mode and len(self.points) < 2:
                self.points.append((x, y))
                print(f"점 {len(self.points)} 설정: ({x}, {y})")

                if len(self.points) == 2:
                    self.line_completed = True
                    print("라인 완성! 'S' 키를 눌러 저장하세요.")

    def draw_interface(self, frame: np.ndarray) -> np.ndarray:
        """인터페이스 그리기"""
        display_frame = frame.copy()

        # 기존에 설정된 점들 그리기
        for i, point in enumerate(self.points):
            cv2.circle(display_frame, point, 8, (0, 255, 0), -1)  # 초록색 점
            cv2.putText(
                display_frame,
                f"P{i+1}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # 라인 그리기
        if len(self.points) == 2:
            # 완성된 라인
            cv2.line(
                display_frame,
                self.points[0],
                self.points[1],
                self.line_color,
                self.line_thickness,
            )
        elif len(self.points) == 1:
            # 미리보기 라인 (첫 번째 점에서 마우스까지)
            cv2.line(
                display_frame,
                self.points[0],
                self.mouse_pos,
                (128, 128, 128),  # 회색 미리보기
                2,
            )

        # 상태 정보 표시
        self.draw_status_panel(display_frame)

        return display_frame

    def draw_status_panel(self, frame: np.ndarray):
        """상태 패널 그리기 (배경 없이 깔끔하게)"""
        img_h, img_w = frame.shape[:2]

        # 상태 정보
        status_lines = [
            f"Sensor: {self.current_sensor_sn or 'Detecting...'}",
            f"Points: {len(self.points)}/2",
        ]

        # 키 조작 안내 (상황별로)
        if len(self.points) == 0:
            status_lines.append("Click: First point")
        elif len(self.points) == 1:
            status_lines.append("Click: Second point")
        elif len(self.points) == 2:
            status_lines.append("Press 'S': Save line")

        # 하단에 키 조작 안내
        key_help = ["Keys: S=Save  R=Reset  ESC=Exit"]

        # 좌상단에 상태 정보 표시 (배경 없이)
        for i, line in enumerate(status_lines):
            y = 25 + i * 20
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # 노란색
                1,
            )

        # 좌하단에 키 조작 안내 표시
        for i, line in enumerate(key_help):
            y = img_h - 15 - (len(key_help) - i - 1) * 20
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),  # 흰색
                1,
            )

    def save_line_config(self) -> bool:
        """라인 설정을 JSON 파일로 저장 (Multi-Sensor 지원)"""
        if len(self.points) != 2:
            print("Error: 라인이 완성되지 않았습니다. 두 개의 점을 설정해주세요.")
            return False

        if not self.current_sensor_sn:
            print("Error: 현재 센서 정보를 확인할 수 없습니다.")
            return False

        try:
            # 설정 파일 디렉토리 생성
            config_path = Path(self.config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # 기존 설정 로드
            config_data = {"version": "1.0", "sensors": {}, "metadata": {}}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

            # 현재 센서의 라인 데이터 생성
            line_data = {
                "name": self.line_name,
                "start_point": list(self.points[0]),
                "end_point": list(self.points[1]),
                "is_active": True,
                "view_type": "entrance",
                "thickness": self.line_thickness,
                "color": list(self.line_color),
            }

            # 센서별 설정 업데이트
            if "sensors" not in config_data:
                config_data["sensors"] = {}

            config_data["sensors"][self.current_sensor_sn] = {
                "sensor_sn": self.current_sensor_sn,
                "line": line_data,
            }

            # 메타데이터 업데이트
            config_data["metadata"] = {
                "created_at": config_data.get("metadata", {}).get(
                    "created_at", datetime.now().isoformat()
                ),
                "updated_at": datetime.now().isoformat(),
                "total_sensors": len(config_data["sensors"]),
                "current_sensor": self.current_sensor_sn,
            }

            # JSON 파일 저장
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            print(f"센서 {self.current_sensor_sn} 라인 설정 저장 완료: {config_path}")
            print(f"  - {self.line_name}: {self.points[0]} -> {self.points[1]}")
            return True

        except Exception as e:
            print(f"라인 설정 저장 실패: {e}")
            return False

    def reset_line(self):
        """라인 초기화"""
        self.points.clear()
        self.line_completed = False
        self.drawing_mode = True
        print("라인 초기화 완료")

    def run(self):
        """메인 실행 루프"""
        try:
            # OpenCV 윈도우 생성
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            print("라인 그리기 도구 시작...")
            print("라이다 센서 연결 중...")

            while True:
                # 라이다 데이터 수신
                lidar_images = self.lidar_receiver.receive_data()

                # 현재 센서 정보 업데이트
                self.update_current_sensor()

                if lidar_images and self.lidar_receiver.sensor_sn:
                    # 첫 번째 센서의 이미지 가져오기
                    sensor_id = self.lidar_receiver.sensor_sn[0]
                    frame = lidar_images.get(sensor_id)

                    if frame is not None:
                        # 인터페이스 그리기
                        display_frame = self.draw_interface(frame)

                        # 화면 표시
                        cv2.imshow(self.window_name, display_frame)
                    else:
                        print("Warning: 라이다 프레임 데이터가 없습니다.")
                else:
                    # 라이다 데이터가 없는 경우 검은 화면 표시
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        dummy_frame,
                        "Connecting to Lidar...",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow(self.window_name, dummy_frame)

                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC - 종료
                    break
                elif key == ord("s") or key == ord("S"):  # S - 저장
                    if self.save_line_config():
                        print("✓ 라인 저장 완료! (계속 그리거나 ESC로 종료)")
                elif key == ord("r") or key == ord("R"):  # R - 초기화
                    self.reset_line()

        except KeyboardInterrupt:
            print("\n사용자가 중단했습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """정리 작업"""
        cv2.destroyAllWindows()
        print("라인 그리기 도구 종료")


def main():
    """메인 함수"""
    config_path = "configs/line_configs.json"

    # 라인 그리기 도구 실행
    drawer = LineDrawer(config_path)
    drawer.run()


if __name__ == "__main__":
    main()
