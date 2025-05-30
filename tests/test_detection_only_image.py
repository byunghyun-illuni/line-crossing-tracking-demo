#!/usr/bin/env python3
"""
Detection만 테스트하는 스크립트 - data/people_1.png 고정 테스트
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


def test_detection_on_image_fixed():
    """data/people_1.png에서 고정 detection 테스트"""

    print("🖼️  고정 이미지 Detection 테스트 (data/people_1.png)")
    print("=" * 60)

    # 이미지 경로 고정
    image_path = "data/people_1.png"

    if not Path(image_path).exists():
        print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
        return

    # 현재 설정으로 테스트 (crowded_scene과 동일)
    config = get_config("crowded_scene")
    detector = YOLOXDetector(
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,  # 0.25
        enable_image_enhancement=False,
        nms_iou_threshold=0.4,
    )

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return

    print(f"📊 이미지 크기: {image.shape}")
    print(f"🔧 사용 설정: {config.model_name}, 임계값: {config.confidence_threshold}")

    # Detection 수행
    start_time = time.time()
    detections = detector.detect_objects(image)
    detection_time = time.time() - start_time

    print(f"⏱️  Detection 시간: {detection_time:.3f}초")
    print(f"👁️  감지된 객체 수: {len(detections)}")

    # 신뢰도별 분류
    high_conf = [d for d in detections if d.confidence >= 0.7]
    medium_conf = [d for d in detections if 0.4 <= d.confidence < 0.7]
    low_conf = [d for d in detections if d.confidence < 0.4]

    print(f"📊 신뢰도별 분류:")
    print(f"   높음 (≥0.7): {len(high_conf)}개")
    print(f"   중간 (0.4~0.7): {len(medium_conf)}개")
    print(f"   낮음 (<0.4): {len(low_conf)}개")

    # 모든 감지 결과 표시 (상위 20개)
    detections_sorted = sorted(detections, key=lambda x: x.confidence, reverse=True)
    print("\n📋 상위 20개 감지 결과:")
    for i, det in enumerate(detections_sorted[:20]):
        print(
            f"   {i+1:2d}. 신뢰도: {det.confidence:.3f}, 크기: {det.bbox[2]:3d}x{det.bbox[3]:3d}, 위치: ({det.bbox[0]:4d}, {det.bbox[1]:4d})"
        )

    # 바운딩 박스 유효성 체크
    img_h, img_w = image.shape[:2]
    valid_detections = []
    invalid_detections = []

    for det in detections:
        x, y, w, h = det.bbox
        x1, y1, x2, y2 = x, y, x + w, y + h

        # 이미지 경계 체크
        if x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h and w > 0 and h > 0:
            valid_detections.append(det)
        else:
            invalid_detections.append(det)

    print(f"\n🔍 바운딩 박스 유효성 체크:")
    print(f"   유효한 박스: {len(valid_detections)}개")
    print(f"   무효한 박스: {len(invalid_detections)}개")

    if invalid_detections:
        print("❌ 무효한 박스 예시:")
        for i, det in enumerate(invalid_detections[:5]):
            x, y, w, h = det.bbox
            print(
                f"   {i+1}. 위치: ({x}, {y}), 크기: {w}x{h}, 신뢰도: {det.confidence:.3f}"
            )

    # 결과를 이미지에 그리기 - 유효한 감지 결과만
    result_image = image.copy()

    print(f"\n🎨 {len(valid_detections)}개 유효한 감지 결과를 시각화 중...")

    drawn_count = 0
    for i, det in enumerate(valid_detections):
        x, y, w, h = det.bbox

        # 다시 한번 경계 체크
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            continue

        # 신뢰도에 따른 색상과 두께
        if det.confidence >= 0.7:
            color = (0, 255, 0)  # 초록색 - 높은 신뢰도
            thickness = 3
        elif det.confidence >= 0.4:
            color = (0, 255, 255)  # 노란색 - 중간 신뢰도
            thickness = 2
        else:
            color = (0, 0, 255)  # 빨간색 - 낮은 신뢰도
            thickness = 1

        # 바운딩 박스 그리기
        try:
            cv2.rectangle(
                result_image,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                color,
                thickness,
            )
            drawn_count += 1

            # 신뢰도 라벨 (작게)
            label = f"{det.confidence:.2f}"
            font_scale = 0.4
            font_thickness = 1

            # 라벨 크기 계산
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # 라벨 위치 조정 (이미지 경계 고려)
            label_x = max(0, min(int(x), img_w - label_w))
            label_y = max(label_h + 5, int(y))

            # 라벨 배경
            cv2.rectangle(
                result_image,
                (label_x, label_y - label_h - 2),
                (label_x + label_w, label_y + 2),
                color,
                -1,
            )

            # 라벨 텍스트
            cv2.putText(
                result_image,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        except Exception as e:
            print(f"⚠️  바운딩 박스 그리기 실패 (인덱스 {i}): {e}")
            continue

    print(f"✅ 실제로 그려진 바운딩 박스: {drawn_count}개")

    # 정보 텍스트 추가
    info_texts = [
        f"Total detections: {len(detections)}",
        f"Valid detections: {len(valid_detections)}",
        f"Drawn boxes: {drawn_count}",
        f"High conf (>=0.7): {len(high_conf)}",
        f"Medium conf (0.4-0.7): {len(medium_conf)}",
        f"Low conf (<0.4): {len(low_conf)}",
    ]

    for i, text in enumerate(info_texts):
        cv2.putText(
            result_image,
            text,
            (10, 25 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 범례 추가 (더 아래쪽으로)
    legend_y = img_h - 100
    cv2.putText(
        result_image,
        "Legend:",
        (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    # 범례 박스들
    legend_items = [
        ("High conf (>=0.7)", (0, 255, 0)),
        ("Medium conf (0.4-0.7)", (0, 255, 255)),
        ("Low conf (<0.4)", (0, 0, 255)),
    ]

    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + 15 + i * 20
        cv2.rectangle(result_image, (10, y_pos), (25, y_pos + 10), color, -1)
        cv2.putText(
            result_image,
            text,
            (30, y_pos + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

    # 결과 저장
    output_path = "temp/detection_result_debug.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"💾 결과 이미지 저장: {output_path}")

    # 화면에 표시
    cv2.imshow("Debug - All Detections", result_image)
    print("🖱️  아무 키나 누르면 종료...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_detection_on_video(video_path: str):
    """비디오에서 detection 테스트"""

    print("🎯 Detection 테스트 시작...")

    # 1. YOLOXDetector 초기화 - 개선된 설정
    print("📡 YOLOX Detector 초기화 중...")
    try:
        detector = YOLOXDetector(
            model_name="fasterrcnn_resnet50_fpn",
            confidence_threshold=0.7,  # 더 높은 임계값
            enable_image_enhancement=False,  # 비디오에서는 비활성화
            nms_iou_threshold=0.4,  # 더 엄격한 NMS
        )
        print(f"✅ Detector 초기화 완료")
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


if __name__ == "__main__":
    print("🚀 YOLOX Detection 디버그 테스트 시작")
    print("=" * 50)

    # 고정 이미지 테스트만 실행
    test_detection_on_image_fixed()
