#!/usr/bin/env python3
"""
OC-SORT 추적 시스템 테스트 스크립트 - data/people.mp4 사용
실제 비디오 데이터로 ID 트래킹 테스트
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


def test_tracking_on_video_frames(
    video_path: str, frame_indices=[50, 100, 150, 200, 250]
):
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
        det_thresh=0.25,  # 0.6 -> 0.25로 낮춤 (더 많은 detection 허용)
        max_age=30,  # 최대 추적 유지 프레임
        min_hits=3,  # 추적 시작 최소 hit 수
        iou_threshold=0.3,  # IoU threshold for association
        delta_t=3,  # velocity calculation delta
        asso_func="iou",  # association function
        inertia=0.2,  # velocity inertia
        use_byte=False,  # ByteTrack association
        detector_config=config,  # detector config
        enable_image_enhancement=False,
        nms_iou_threshold=0.3,
    )

    print(
        f"🎯 Tracker 초기화: {config.model_name}, 검출임계값: {config.confidence_threshold}, 추적임계값: 0.25"
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
            else:
                color = (128, 128, 128)  # 회색 - ID 없음
                thickness = 1

            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            drawn_count += 1

            # Track ID와 신뢰도 라벨
            if track_id > 0:
                label = f"ID{track_id}: {det.confidence:.2f}"
            else:
                label = f"?: {det.confidence:.2f}"

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


def test_continuous_tracking(video_path: str, start_frame=100, num_frames=50):
    """연속 프레임에서 tracking 테스트 - ID 일관성 확인"""

    print(f"\n🔄 연속 추적 테스트 (프레임 {start_frame}~{start_frame+num_frames-1})")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다")
        return

    # Tracker 초기화
    config = get_config("crowded_scene")
    tracker = ObjectTracker(
        det_thresh=0.25,  # 0.6 -> 0.25로 낮춤
        max_age=10,  # 더 짧은 max_age로 빠른 테스트
        min_hits=2,  # 더 빠른 트랙 생성
        iou_threshold=0.3,
        detector_config=config,
        enable_image_enhancement=False,
    )

    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ID 추적 통계
    id_history = {}  # track_id: [frame_numbers]
    frame_stats = []

    print("📹 연속 프레임 추적 시작...")

    for i in range(num_frames):
        current_frame = start_frame + i
        ret, frame = cap.read()
        if not ret:
            break

        # Tracking 수행
        tracking_frame = tracker.track_frame(frame)

        # Track ID 기록
        current_ids = set()
        for det in tracking_frame.detections:
            track_id = getattr(det, "track_id", -1)
            if track_id > 0:
                current_ids.add(track_id)
                if track_id not in id_history:
                    id_history[track_id] = []
                id_history[track_id].append(current_frame)

        # 프레임별 통계
        frame_stats.append(
            {
                "frame": current_frame,
                "total_tracks": len(tracking_frame.detections),
                "valid_ids": len(current_ids),
                "ids": sorted(current_ids),
            }
        )

        # 10프레임마다 출력
        if i % 10 == 0 or i < 5:
            print(
                f"   프레임 {current_frame}: {len(tracking_frame.detections)}개 추적, ID: {sorted(current_ids)}"
            )

    cap.release()

    # ID 추적 분석 결과
    print("\n📊 연속 추적 분석 결과:")
    print(f"   처리된 프레임: {len(frame_stats)}")
    print(f"   총 고유 ID 수: {len(id_history)}")

    # ID별 지속성 분석
    print("\n🆔 ID별 추적 지속성:")
    for track_id, frames in sorted(id_history.items()):
        duration = len(frames)
        first_frame = frames[0]
        last_frame = frames[-1]
        print(
            f"   ID {track_id:2d}: {duration:2d}프레임 지속 (프레임 {first_frame}~{last_frame})"
        )

    # 평균 통계
    avg_tracks = sum(stat["total_tracks"] for stat in frame_stats) / len(frame_stats)
    avg_valid_ids = sum(stat["valid_ids"] for stat in frame_stats) / len(frame_stats)

    print("\n📈 평균 통계:")
    print(f"   평균 추적 객체 수: {avg_tracks:.1f}개")
    print(f"   평균 유효 ID 수: {avg_valid_ids:.1f}개")


if __name__ == "__main__":
    video_path = "data/people.mp4"

    if not Path(video_path).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        print("📁 data 디렉토리의 파일들:")
        for f in Path("data").glob("*.mp4"):
            print(f"   {f}")
        exit(1)

    print("🚀 OC-SORT 추적 시스템 테스트 시작")
    print("=" * 50)

    # temp 디렉토리 생성
    Path("temp").mkdir(exist_ok=True)

    # 1. 특정 프레임들에서 추적 테스트
    test_tracking_on_video_frames(video_path, [50, 100, 150, 200, 287])

    print("\n" + "=" * 50)

    # 2. 연속 프레임에서 ID 일관성 테스트
    test_continuous_tracking(video_path, start_frame=100, num_frames=30)

    print("\n🎯 결론:")
    print("1. temp/tracking_frame_*.jpg 파일들을 확인해보세요")
    print("2. 각 ID별로 고유한 색상으로 바운딩 박스가 그려집니다")
    print("3. 연속 프레임에서 ID 일관성이 유지되는지 확인하세요")
    print("4. 추적 지속성 분석 결과를 참고하세요")
