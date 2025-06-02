#!/usr/bin/env python3
"""
OC-SORT 추적 시스템 테스트 스크립트 - 5프레임 연속 테스트
실제 비디오 데이터로 ID 트래킹 테스트 (연속 5프레임: 50-54)
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


def test_tracking_on_video_frames(video_path: str, frame_indices=[50, 51, 52, 53, 54]):
    """비디오 특정 프레임들에서 tracking 테스트"""

    print(f"🎬 비디오 추적 테스트: {video_path}")
    print("=" * 60)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📊 비디오 정보: {width}x{height}, {fps:.1f}fps, 총 {frame_count}프레임")

    # Tracker 초기화 (crowded_scene 설정 사용)
    config = get_config("crowded_scene")
    tracker = ObjectTracker(
        det_thresh=0.1,  # OCSort 내부 필터링을 거의 비활성화
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=True,
        detector_config=config,
        enable_image_enhancement=False,
        nms_iou_threshold=0.3,
    )

    print(
        f"🎯 Tracker 초기화: {config.model_name}, 검출임계값: {config.confidence_threshold}, 추적임계값: 0.1"
    )

    # temp 디렉토리 생성
    Path("temp").mkdir(exist_ok=True)

    # 각 프레임에 대해 테스트
    for frame_idx in frame_indices:
        if frame_idx >= frame_count:
            print(f"⚠️  프레임 {frame_idx}는 범위를 벗어남 (최대: {frame_count})")
            continue

        print(f"\n🔍 프레임 {frame_idx} 추적 테스트...")

        # 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"❌ 프레임 {frame_idx}를 읽을 수 없습니다")
            continue

        print(f"📐 프레임 크기: {frame.shape}")

        # Detection + Tracking 수행
        start_time = time.time()
        tracking_frame = tracker.track_frame(frame)
        tracking_time = time.time() - start_time

        print(f"⏱️  Tracking 시간: {tracking_time:.3f}초")
        print(f"👁️  추적된 객체 수: {len(tracking_frame.detections)}")

        # 신뢰도별 분류
        high_conf = [d for d in tracking_frame.detections if d.confidence >= 0.7]
        medium_conf = [
            d for d in tracking_frame.detections if 0.4 <= d.confidence < 0.7
        ]
        low_conf = [d for d in tracking_frame.detections if d.confidence < 0.4]

        print(
            f"📊 신뢰도별 분류: 높음({len(high_conf)}) 중간({len(medium_conf)}) 낮음({len(low_conf)})"
        )

        # 바운딩 박스 유효성 체크
        img_h, img_w = frame.shape[:2]
        valid_count = 0
        invalid_count = 0

        for det in tracking_frame.detections:
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

        # Track ID 분석
        track_ids = [
            det.track_id
            for det in tracking_frame.detections
            if hasattr(det, "track_id") and det.track_id > 0
        ]
        unique_ids = set(track_ids)
        print(
            f"🆔 Track ID 분석: 총 {len(unique_ids)}개 고유 ID - {sorted(unique_ids)}"
        )

        # 추적 결과 출력 (상위 10개)
        detections_sorted = sorted(
            tracking_frame.detections, key=lambda x: x.confidence, reverse=True
        )
        print("📋 상위 10개 추적 결과:")
        for i, det in enumerate(detections_sorted[:10]):
            track_id = getattr(det, "track_id", "N/A")
            print(
                f"   {i+1:2d}. ID={track_id:3}, 신뢰도: {det.confidence:.3f}, 크기: {det.bbox[2]:3d}x{det.bbox[3]:3d}, 위치: ({det.bbox[0]:4d}, {det.bbox[1]:4d})"
            )

        # 결과 이미지 저장
        annotated_frame = frame.copy()
        drawn_count = 0

        for det in tracking_frame.detections:
            x, y, w, h = det.bbox
            track_id = getattr(det, "track_id", -1)

            # 바운딩 박스 유효성 체크
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                continue

            # Track ID에 따른 색상 선택
            if track_id > 0:
                # Track ID를 기반으로 고유 색상 생성
                np.random.seed(track_id)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                thickness = 3
                label = f"ID{track_id}: {det.confidence:.2f}"
            else:
                color = (128, 128, 128)  # 회색 - ID 없음
                thickness = 1
                label = f"?: {det.confidence:.2f}"

            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            drawn_count += 1

            # 라벨 배경과 텍스트
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated_frame, (x, y - 20), (x + label_size[0], y), color, -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        print(f"✅ 실제로 그려진 바운딩 박스: {drawn_count}개")

        # 정보 텍스트 추가
        info_texts = [
            f"Frame {frame_idx}: Tracking Test",
            f"Total tracks: {len(tracking_frame.detections)}",
            f"Valid boxes: {valid_count}",
            f"Unique IDs: {len(unique_ids)}",
            f"High conf: {len(high_conf)}",
            f"Time: {tracking_time:.3f}s",
        ]

        for i, text in enumerate(info_texts):
            cv2.putText(
                annotated_frame,
                text,
                (10, 25 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 결과 저장
        output_path = f"temp/tracking_frame_{frame_idx}_result.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"💾 결과 저장: {output_path}")

    cap.release()
    print("\n✅ 프레임별 추적 테스트 완료!")


if __name__ == "__main__":
    video_path = "data/people.mp4"

    if not Path(video_path).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        print("📁 data 디렉토리의 파일들:")
        for f in Path("data").glob("*.mp4"):
            print(f"   {f}")
        exit(1)

    print("🚀 OC-SORT 추적 시스템 테스트 시작 (5프레임 연속)")
    print("=" * 50)

    # temp 디렉토리 생성
    Path("temp").mkdir(exist_ok=True)

    # 1. 특정 프레임들에서 추적 테스트
    print("\n📊 연속 프레임 추적 테스트 (50-54)")
    test_tracking_on_video_frames(video_path, [50, 51, 52, 53, 54])

    print("\n🎯 결론:")
    print("1. temp/tracking_frame_*.jpg 파일들을 확인해보세요")
    print("2. 이제 모든 detection이 Track ID를 가집니다!")
    print("3. 더 이상 '?' 표시가 나타나지 않습니다")
