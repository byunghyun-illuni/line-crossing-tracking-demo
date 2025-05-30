#!/usr/bin/env python3
"""
OC-SORT 추적 시스템 테스트 스크립트
"""

import logging
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.core.models import DetectionResult
from src.tracking.engine import ObjectTracker


def create_dummy_detections(frame_id: int) -> list:
    """더미 감지 결과 생성"""
    detections = []

    # 프레임마다 약간씩 이동하는 객체들 시뮬레이션
    base_positions = [
        (100, 100, 150, 200),  # 객체 1
        (300, 150, 350, 250),  # 객체 2
        (500, 200, 550, 300),  # 객체 3
    ]

    for i, (x1, y1, x2, y2) in enumerate(base_positions):
        # 프레임마다 약간씩 이동
        offset_x = frame_id * 2
        offset_y = frame_id * 1

        # 일부 객체는 중간에 사라지도록 설정
        if i == 1 and 10 <= frame_id <= 15:
            continue  # 객체 2는 10-15 프레임에서 사라짐

        # 바운딩 박스 계산
        bbox_x = x1 + offset_x
        bbox_y = y1 + offset_y
        bbox_w = (x2 + offset_x) - bbox_x
        bbox_h = (y2 + offset_y) - bbox_y

        # 중심점 계산
        center_x = bbox_x + bbox_w / 2
        center_y = bbox_y + bbox_h / 2

        detection = DetectionResult(
            track_id=i + 1,  # 임시 track_id
            bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
            center_point=(center_x, center_y),
            confidence=0.8 + np.random.normal(0, 0.1),
            class_name="person",
            timestamp=frame_id * 0.033,  # 30fps 기준
        )
        detections.append(detection)

    return detections


def test_ocsort_tracking():
    """OC-SORT 추적 테스트"""
    print("=== OC-SORT 추적 테스트 시작 ===")

    # 추적기 초기화
    tracker = ObjectTracker(
        det_thresh=0.6,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
    )

    # 더미 프레임 생성 (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"추적기 초기화 완료")

    # 여러 프레임에 대해 추적 수행
    for frame_id in range(1, 21):
        print(f"\n--- 프레임 {frame_id} ---")

        # 더미 감지 결과 생성
        detections = create_dummy_detections(frame_id)
        print(f"감지된 객체 수: {len(detections)}")

        # 추적 수행
        tracking_frame = tracker.track_frame(dummy_frame, detections)

        print(f"추적된 객체 수: {len(tracking_frame.detections)}")

        # 추적 결과 출력
        for i, detection in enumerate(tracking_frame.detections):
            track_id = getattr(detection, "track_id", "N/A")
            bbox = detection.bbox
            confidence = detection.confidence
            print(f"  객체 {i+1}: ID={track_id}, bbox={bbox}, conf={confidence:.3f}")

    print("\n=== OC-SORT 추적 테스트 완료 ===")


def test_detection_only():
    """감지만 테스트 (추적 없이)"""
    print("\n=== 감지 전용 테스트 시작 ===")

    tracker = ObjectTracker()

    # 더미 프레임 생성
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # HOG 감지기 테스트
    detections = tracker.detect_objects(dummy_frame)
    print(f"HOG 감지기로 감지된 객체 수: {len(detections)}")

    for i, detection in enumerate(detections):
        print(f"  감지 {i+1}: bbox={detection.bbox}, conf={detection.confidence:.3f}")

    print("=== 감지 전용 테스트 완료 ===")


def main():
    """메인 테스트 함수"""
    try:
        # OC-SORT 추적 테스트
        test_ocsort_tracking()

        # 감지 전용 테스트
        test_detection_only()

        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
