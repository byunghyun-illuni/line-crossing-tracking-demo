#!/usr/bin/env python3
"""
비디오 프레임 회전 문제 진단 및 수정
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def diagnose_video_orientation(video_path: str):
    """비디오 방향 문제 진단"""

    print(f"🔍 비디오 방향 진단: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다")
        return

    # 비디오 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 비디오 메타데이터:")
    print(f"   해상도: {width} x {height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {frame_count}")

    # 회전 정보 확인 (메타데이터)
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    print(f"   회전 메타데이터: {rotation}")

    # 첫 번째 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("❌ 첫 번째 프레임을 읽을 수 없습니다")
        cap.release()
        return

    actual_shape = frame.shape
    print(f"📐 실제 프레임 모양: {actual_shape}")
    print(
        f"   Height x Width x Channels: {actual_shape[0]} x {actual_shape[1]} x {actual_shape[2]}"
    )

    # 예상과 실제 비교
    if actual_shape[0] != height or actual_shape[1] != width:
        print(f"⚠️  불일치 발견!")
        print(f"   메타데이터: {width}x{height}")
        print(f"   실제 프레임: {actual_shape[1]}x{actual_shape[0]}")

    # 프레임 저장 (원본)
    cv2.imwrite("frame_original.jpg", frame)
    print(f"💾 원본 프레임 저장: frame_original.jpg")

    # 90도씩 회전하여 테스트
    angles = [0, 90, 180, 270]
    for angle in angles:
        if angle == 0:
            rotated = frame.copy()
        elif angle == 90:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        filename = f"frame_rotated_{angle}.jpg"
        cv2.imwrite(filename, rotated)
        print(f"🔄 {angle}도 회전 프레임 저장: {filename} (크기: {rotated.shape})")

    cap.release()

    print(f"\n🎯 결론:")
    print(f"1. frame_original.jpg와 frame_rotated_*.jpg 파일들을 확인하세요")
    print(f"2. 올바른 방향을 찾아서 알려주세요")
    print(f"3. Detection은 올바르게 회전된 프레임에서 테스트해야 합니다")


def test_rotated_detection(video_path: str, rotation_angle: int = 0):
    """회전된 프레임으로 detection 테스트"""

    print(f"🎯 회전된 프레임 Detection 테스트 (회전각: {rotation_angle}도)")

    # 프로젝트 루트를 Python 경로에 추가
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from src.tracking.detector_configs import get_config
    from src.tracking.yolox_detector import YOLOXDetector

    # Detector 초기화
    config = get_config("crowded_scene")
    detector = YOLOXDetector(
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,
        enable_image_enhancement=True,
        nms_iou_threshold=0.3,
    )

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다")
        return

    # 특정 프레임으로 이동 (100번째)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()

    if not ret:
        print(f"❌ 프레임을 읽을 수 없습니다")
        cap.release()
        return

    print(f"📐 원본 프레임 크기: {frame.shape}")

    # 회전 적용
    if rotation_angle == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print(f"📐 회전 후 프레임 크기: {frame.shape}")

    # Detection 수행
    import time

    start_time = time.time()
    detections = detector.detect_objects(frame)
    detection_time = time.time() - start_time

    print(f"⏱️  Detection 시간: {detection_time:.3f}초")
    print(f"👁️  감지된 객체 수: {len(detections)}")

    # 결과 그리기
    result_frame = frame.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 라벨
        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(
            result_frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # 처음 10개만 출력
        if i < 10:
            print(
                f"   {i+1}. 클래스: {det.class_name}, 신뢰도: {det.confidence:.3f}, "
                f"위치: {det.center_point}, 바운딩박스: {det.bbox}"
            )

    # 결과 저장
    output_file = f"detection_rotated_{rotation_angle}_result.jpg"
    cv2.imwrite(output_file, result_frame)
    print(f"💾 결과 저장: {output_file}")

    cap.release()


if __name__ == "__main__":
    video_path = "data/people.mp4"

    if not Path(video_path).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        exit(1)

    print("🚀 비디오 회전 문제 진단 시작")
    print("=" * 50)

    # 1. 비디오 방향 진단
    diagnose_video_orientation(video_path)

    print("\n" + "=" * 50)
    print("각 회전 각도별로 detection 테스트를 실행하시겠습니까?")
    choice = input("y/n: ").strip().lower()

    if choice == "y":
        for angle in [0, 90, 180, 270]:
            print(f"\n🔄 {angle}도 회전 테스트:")
            test_rotated_detection(video_path, angle)

    print(f"\n🎯 다음 단계:")
    print(f"1. frame_*.jpg 파일들을 확인하여 올바른 방향을 찾으세요")
    print(f"2. detection_rotated_*_result.jpg 파일들을 비교하세요")
    print(f"3. 가장 좋은 결과를 보이는 회전각을 알려주세요")
