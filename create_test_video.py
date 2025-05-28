"""
테스트용 비디오 생성 스크립트
움직이는 객체들이 있는 간단한 비디오를 생성합니다.
"""

import os

import cv2
import numpy as np


def create_test_video(output_path="test_video.mp4", duration_seconds=10, fps=30):
    """
    테스트용 비디오 생성

    Args:
        output_path: 출력 비디오 파일 경로
        duration_seconds: 비디오 길이 (초)
        fps: 프레임 레이트
    """
    # 비디오 설정
    width, height = 640, 480
    total_frames = duration_seconds * fps

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 움직이는 객체들 설정
    objects = [
        {"pos": [50, 100], "vel": [2, 1], "size": 30, "color": (0, 255, 0)},  # 녹색 원
        {
            "pos": [200, 200],
            "vel": [-1, 2],
            "size": 40,
            "color": (255, 0, 0),
        },  # 파란색 원
        {
            "pos": [400, 150],
            "vel": [1, -1],
            "size": 35,
            "color": (0, 0, 255),
        },  # 빨간색 원
    ]

    print(f"테스트 비디오 생성 중... ({total_frames} 프레임)")

    for frame_idx in range(total_frames):
        # 검은 배경 생성
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 프레임 번호 표시
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # 각 객체 그리기 및 이동
        for obj in objects:
            # 현재 위치에 원 그리기
            cv2.circle(
                frame,
                (int(obj["pos"][0]), int(obj["pos"][1])),
                obj["size"],
                obj["color"],
                -1,
            )

            # 객체 이동
            obj["pos"][0] += obj["vel"][0]
            obj["pos"][1] += obj["vel"][1]

            # 경계에서 반사
            if obj["pos"][0] <= obj["size"] or obj["pos"][0] >= width - obj["size"]:
                obj["vel"][0] *= -1
            if obj["pos"][1] <= obj["size"] or obj["pos"][1] >= height - obj["size"]:
                obj["vel"][1] *= -1

            # 경계 내로 위치 조정
            obj["pos"][0] = max(obj["size"], min(width - obj["size"], obj["pos"][0]))
            obj["pos"][1] = max(obj["size"], min(height - obj["size"], obj["pos"][1]))

        # 가상 라인 그리기 (테스트용)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 2)
        cv2.putText(
            frame,
            "Virtual Line",
            (width // 2 + 10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # 프레임 저장
        out.write(frame)

        # 진행률 표시
        if frame_idx % (total_frames // 10) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"진행률: {progress:.1f}%")

    # 리소스 해제
    out.release()
    print(f"테스트 비디오가 생성되었습니다: {output_path}")
    print(f"파일 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    create_test_video("test_video.mp4", duration_seconds=15, fps=30)
    print("\n사용법:")
    print("1. Streamlit 앱을 실행: streamlit run streamlit_app/main.py")
    print("2. '비디오 파일 업로드' 선택")
    print("3. 생성된 test_video.mp4 파일 업로드")
    print("4. 트래커 초기화 후 추적 시작")
