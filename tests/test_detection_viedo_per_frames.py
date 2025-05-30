#!/usr/bin/env python3
"""
비디오 프레임 추출 및 Detection 테스트
people.mp4의 특정 프레임들을 추출해서 개별적으로 테스트
"""

import sys
import time
from pathlib import Path

import cv2

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tracking.detector_configs import get_config
from src.tracking.yolox_detector import YOLOXDetector


def extract_and_test_frames(video_path: str, frame_indices=[10, 50, 100, 200, 300]):
    """비디오에서 특정 프레임들을 추출해서 detection 테스트"""

    print(f"🎬 비디오 프레임 추출 테스트: {video_path}")

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 비디오 정보: {width}x{height}, {fps}fps, 총 {frame_count}프레임")

    # Detector 초기화
    config = get_config("crowded_scene")
    detector = YOLOXDetector(
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,
        enable_image_enhancement=True,
        nms_iou_threshold=0.3,
    )
    print(
        f"🎯 Detector 초기화: {config.model_name}, 임계값: {config.confidence_threshold}"
    )

    # 각 프레임에 대해 테스트
    for frame_idx in frame_indices:
        if frame_idx >= frame_count:
            print(f"⚠️  프레임 {frame_idx}는 범위를 벗어남 (최대: {frame_count})")
            continue

        print(f"\n🔍 프레임 {frame_idx} 테스트 중...")

        # 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"❌ 프레임 {frame_idx}를 읽을 수 없습니다")
            continue

        print(f"📐 프레임 크기: {frame.shape}")

        # Detection 수행
        start_time = time.time()
        detections = detector.detect_objects(frame)
        detection_time = time.time() - start_time

        print(f"⏱️  Detection 시간: {detection_time:.3f}초")
        print(f"👁️  감지된 객체 수: {len(detections)}")

        # 신뢰도별 분류
        high_conf = [d for d in detections if d.confidence >= 0.7]
        medium_conf = [d for d in detections if 0.4 <= d.confidence < 0.7]
        low_conf = [d for d in detections if d.confidence < 0.4]

        print(
            f"📊 신뢰도별 분류: 높음({len(high_conf)}) 중간({len(medium_conf)}) 낮음({len(low_conf)})"
        )

        # 바운딩 박스 유효성 체크
        img_h, img_w = frame.shape[:2]
        valid_count = 0
        invalid_count = 0

        for det in detections:
            x, y, w, h = det.bbox
            if (
                x >= 0
                and y >= 0
                and x + w <= img_w
                and y + h <= img_h
                and w > 0
                and h > 0
            ):
                valid_count += 1
            else:
                invalid_count += 1

        print(f"🔍 바운딩 박스: 유효({valid_count}) 무효({invalid_count})")

        # 감지된 객체 정보 출력 (상위 10개만)
        detections_sorted = sorted(detections, key=lambda x: x.confidence, reverse=True)
        print("📋 상위 10개 감지 결과:")
        for i, det in enumerate(detections_sorted[:10]):
            print(
                f"   {i+1:2d}. 신뢰도: {det.confidence:.3f}, 크기: {det.bbox[2]:3d}x{det.bbox[3]:3d}, 위치: ({det.bbox[0]:4d}, {det.bbox[1]:4d})"
            )

        # 결과 이미지 저장
        annotated_frame = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # 프레임 정보 추가
        info_text = (
            f"Frame {frame_idx}: {len(detections)} detections in {detection_time:.3f}s"
        )
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 결과 저장
        output_path = f"temp/debug_frame_{frame_idx}_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"💾 결과 저장: {output_path}")

    cap.release()
    print("\n✅ 테스트 완료!")


def compare_continuous_vs_jump_frames(video_path: str):
    """연속 프레임 vs 점프 프레임 비교"""

    print("🔄 연속 vs 점프 프레임 비교 테스트")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다")
        return

    # Detector 초기화
    config = get_config("crowded_scene")
    detector = YOLOXDetector(
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,
        enable_image_enhancement=True,
    )

    print("\n📹 연속 프레임 읽기 테스트 (프레임 100-110)")
    total_detections_continuous = 0

    # 연속으로 프레임 읽기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_objects(frame)
        total_detections_continuous += len(detections)
        print(f"   프레임 {100+i}: {len(detections)}개 감지")

    print(f"📊 연속 읽기 총 감지: {total_detections_continuous}개")

    print("\n🦘 점프 프레임 읽기 테스트 (100, 200, 300...)")
    total_detections_jump = 0

    # 점프해서 프레임 읽기
    for frame_idx in [100, 200, 300, 400, 500]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        detections = detector.detect_objects(frame)
        total_detections_jump += len(detections)
        print(f"   프레임 {frame_idx}: {len(detections)}개 감지")

    print(f"📊 점프 읽기 총 감지: {total_detections_jump}개")

    cap.release()


if __name__ == "__main__":
    video_path = "data/people.mp4"

    if not Path(video_path).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        print("📁 data 디렉토리의 파일들:")
        for f in Path("data").glob("*.mp4"):
            print(f"   {f}")
        exit(1)

    print("🚀 비디오 프레임 디버깅 시작")
    print("=" * 50)

    # temp 디렉토리 생성
    Path("temp").mkdir(exist_ok=True)

    # 1. 특정 프레임들 추출해서 테스트
    extract_and_test_frames(video_path, [50, 100, 200, 287, 400])

    print("\n" + "=" * 50)

    # 2. 연속 vs 점프 프레임 비교
    compare_continuous_vs_jump_frames(video_path)

    print("\n🎯 결론:")
    print("1. temp/debug_frame_*.jpg 파일들을 확인해보세요")
    print("2. 연속 읽기 vs 점프 읽기 결과를 비교해보세요")
    print("3. Streamlit에서는 연속 읽기를 하므로 차이가 있을 수 있습니다")
