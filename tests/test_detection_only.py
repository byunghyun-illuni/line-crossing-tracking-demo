#!/usr/bin/env python3
"""
Detection만 테스트하는 스크립트
Streamlit 없이 OpenCV로 빠르게 확인
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


def test_detection_on_video(video_path: str):
    """비디오에서 detection 테스트"""

    print("🎯 Detection 테스트 시작...")

    # 1. YOLOXDetector 초기화
    print("📡 YOLOX Detector 초기화 중...")
    try:
        # crowded_scene 설정 사용
        config = get_config("crowded_scene")
        detector = YOLOXDetector(
            model_name=config.model_name,
            confidence_threshold=config.confidence_threshold,
            enable_image_enhancement=True,  # 이미지 향상 활성화
            nms_iou_threshold=0.3,
        )
        print(
            f"✅ Detector 초기화 완료 - 모델: {config.model_name}, 임계값: {config.confidence_threshold}"
        )
    except Exception as e:
        print(f"❌ Detector 초기화 실패: {e}")
        return

    # 2. 비디오 열기
    print(f"📹 비디오 열기: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 비디오 정보: {width}x{height}, {fps}fps, {frame_count}프레임")

    # 3. 프레임별 detection 테스트
    frame_num = 0
    total_detections = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # 매 10번째 프레임만 처리 (속도 향상)
        if frame_num % 10 != 0:
            continue

        print(f"\n🔍 프레임 {frame_num} 처리 중...")

        # Detection 수행
        detection_start = time.time()
        detections = detector.detect_objects(frame)
        detection_time = time.time() - detection_start

        print(f"⏱️  Detection 시간: {detection_time:.3f}초")
        print(f"👁️  감지된 객체 수: {len(detections)}")

        # 감지된 객체 정보 출력
        for i, det in enumerate(detections):
            print(
                f"   {i+1}. ID: {det.track_id}, 클래스: {det.class_name}, "
                f"신뢰도: {det.confidence:.3f}, 바운딩박스: {det.bbox}"
            )

        total_detections += len(detections)

        # 결과를 프레임에 그리기
        annotated_frame = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 라벨 표시
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

        # 정보 표시
        info_text = f"Frame: {frame_num}, Detections: {len(detections)}, Time: {detection_time:.3f}s"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 화면에 표시
        cv2.imshow("Detection Test", annotated_frame)

        # ESC 키로 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n⏹️  사용자에 의해 중단됨")
            break

        # 처음 50프레임만 처리 (빠른 테스트)
        if frame_num >= 50:
            print(f"\n⏹️  테스트 완료 (50프레임 처리)")
            break

    # 결과 요약
    elapsed_time = time.time() - start_time
    avg_detections = total_detections / (frame_num // 10) if frame_num > 0 else 0

    print(f"\n📈 테스트 결과:")
    print(f"   처리된 프레임: {frame_num // 10}")
    print(f"   총 소요 시간: {elapsed_time:.2f}초")
    print(f"   평균 초당 프레임: {(frame_num // 10) / elapsed_time:.2f} FPS")
    print(f"   총 감지 객체: {total_detections}")
    print(f"   평균 프레임당 감지: {avg_detections:.1f}개")

    cap.release()
    cv2.destroyAllWindows()


def test_detection_on_image(image_path: str):
    """단일 이미지에서 detection 테스트"""

    print("🖼️  단일 이미지 Detection 테스트...")

    # YOLOXDetector 초기화
    config = get_config("crowded_scene")
    detector = YOLOXDetector(
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,
        enable_image_enhancement=True,
    )

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return

    print(f"📊 이미지 크기: {image.shape}")

    # Detection 수행
    start_time = time.time()
    detections = detector.detect_objects(image)
    detection_time = time.time() - start_time

    print(f"⏱️  Detection 시간: {detection_time:.3f}초")
    print(f"👁️  감지된 객체 수: {len(detections)}")

    # 결과 표시
    for i, det in enumerate(detections):
        print(
            f"   {i+1}. 클래스: {det.class_name}, 신뢰도: {det.confidence:.3f}, 바운딩박스: {det.bbox}"
        )

    # 결과를 이미지에 그리기
    result_image = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{det.class_name}: {det.confidence:.2f}"
        cv2.putText(
            result_image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # 결과 저장 및 표시
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"💾 결과 이미지 저장: {output_path}")

    cv2.imshow("Detection Result", result_image)
    print("🖱️  아무 키나 누르면 종료...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("🚀 YOLOX Detection 테스트 시작")
    print("=" * 50)

    # 사용자 입력
    test_type = input("테스트 유형 선택 (1: 비디오, 2: 이미지): ").strip()

    if test_type == "1":
        video_path = input("비디오 파일 경로 입력: ").strip()
        if video_path and Path(video_path).exists():
            test_detection_on_video(video_path)
        else:
            print("❌ 유효한 비디오 파일 경로를 입력해주세요")

    elif test_type == "2":
        image_path = input("이미지 파일 경로 입력: ").strip()
        if image_path and Path(image_path).exists():
            test_detection_on_image(image_path)
        else:
            print("❌ 유효한 이미지 파일 경로를 입력해주세요")
    else:
        print("❌ 올바른 옵션을 선택해주세요 (1 또는 2)")
