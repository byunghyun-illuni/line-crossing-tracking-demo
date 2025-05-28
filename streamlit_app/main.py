"""
2D Access Control MVP - Streamlit 메인 애플리케이션

OC-SORT 기반 라인 크로싱 추적 시스템
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# macOS 카메라 권한 문제 해결을 위한 환경 변수 설정
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.models import DetectionResult
from src.tracking.engine import ObjectTracker
from src.video.source import VideoSource

# 설정
st.set_page_config(
    page_title="2D Access Control System",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="expanded",
)


class StreamlitApp:
    """Streamlit 애플리케이션 클래스"""

    def __init__(self):
        self.tracker = None
        self.video_source = None
        self.is_running = False
        self.frame_count = 0
        self.detection_count = 0
        self.crossing_count = 0

    def initialize_tracker(self, confidence_threshold: float):
        """트래커 초기화"""
        try:
            self.tracker = ObjectTracker(det_thresh=confidence_threshold)
            st.success("트래커가 초기화되었습니다!")
            return True
        except Exception as e:
            st.error(f"트래커 초기화 실패: {e}")
            return False

    def process_video_file(self, uploaded_file, confidence_threshold: float):
        """업로드된 비디오 파일 처리"""
        if uploaded_file is not None:
            try:
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # VideoSource로 비디오 열기
                self.video_source = VideoSource()
                if self.video_source.open(tmp_file_path):
                    frame_count = self.video_source.get_frame_count()
                    fps = self.video_source.get_fps()
                    st.success(
                        f"비디오 파일이 로드되었습니다! (총 {frame_count}프레임, {fps:.1f} FPS)"
                    )
                    return True
                else:
                    st.error("비디오 파일을 열 수 없습니다.")
                    return False
            except Exception as e:
                st.error(f"비디오 파일 처리 중 오류: {e}")
                return False
        return False

    def process_camera(self, camera_id: int):
        """카메라 스트림 처리"""
        try:
            self.video_source = VideoSource()
            if self.video_source.open(camera_id):
                st.success(f"카메라 {camera_id}가 연결되었습니다!")
                return True
            else:
                st.error(
                    f"카메라 {camera_id}를 열 수 없습니다. 다른 카메라 ID를 시도해보세요."
                )
                st.info(
                    "💡 macOS에서는 카메라 권한이 필요할 수 있습니다. 시스템 환경설정 > 보안 및 개인정보보호 > 카메라에서 권한을 확인해주세요."
                )
                return False
        except Exception as e:
            st.error(f"카메라 연결 중 오류: {e}")
            return False

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """프레임에 감지 결과 그리기"""
        for detection in detections:
            x, y, w, h = detection.bbox
            track_id = detection.track_id
            confidence = detection.confidence
            class_name = detection.class_name

            # 바운딩 박스 그리기
            color = (0, 255, 0) if track_id > 0 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 트랙 ID와 정보 표시
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
            cv2.putText(
                frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # 중심점 그리기
            center_x, center_y = detection.center_point
            cv2.circle(frame, (int(center_x), int(center_y)), 3, color, -1)

        return frame

    def run_tracking(self, video_placeholder, stats_placeholder, events_placeholder):
        """트래킹 실행"""
        if self.video_source is None or self.tracker is None:
            st.error("비디오 소스와 트래커를 먼저 초기화해주세요.")
            return

        self.frame_count = 0
        self.detection_count = 0

        try:
            while self.is_running:
                success, frame = self.video_source.read_frame()

                if not success:
                    st.warning("비디오 스트림이 종료되었습니다.")
                    self.is_running = False
                    break

                # 트래킹 수행
                tracking_frame = self.tracker.track_frame(frame)

                # 감지 결과 그리기
                annotated_frame = self.draw_detections(
                    frame.copy(), tracking_frame.detections
                )

                # 프레임 정보 표시
                cv2.putText(
                    annotated_frame,
                    f"Frame: {self.frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                # 비디오 표시 (deprecated 파라미터 수정)
                video_placeholder.image(
                    annotated_frame, channels="BGR", use_container_width=True
                )

                # 통계 업데이트
                self.frame_count += 1
                self.detection_count += len(tracking_frame.detections)

                # 통계 표시
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("프레임", self.frame_count)
                    with col2:
                        st.metric("총 감지", self.detection_count)
                    with col3:
                        st.metric("라인 교차", self.crossing_count)

                # 이벤트 표시
                if tracking_frame.detections:
                    with events_placeholder.container():
                        st.write("**최근 감지된 객체들:**")
                        for det in tracking_frame.detections[-5:]:  # 최근 5개만 표시
                            st.write(
                                f"- ID {det.track_id}: {det.class_name} (신뢰도: {det.confidence:.2f})"
                            )

                # 프레임 레이트 조절
                time.sleep(0.03)  # 약 30 FPS

        except Exception as e:
            st.error(f"트래킹 중 오류 발생: {e}")
            self.is_running = False

    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if self.video_source:
            self.video_source.release()
            self.video_source = None


def main():
    """메인 애플리케이션"""
    st.title("🚪 2D Access Control MVP")
    st.markdown("**OC-SORT 기반 객체 추적 및 라인 크로싱 감지 시스템**")
    st.markdown("---")

    # 세션 상태 초기화
    if "app" not in st.session_state:
        st.session_state.app = StreamlitApp()

    app = st.session_state.app

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")

        # 비디오 소스 선택
        video_source_type = st.selectbox(
            "비디오 소스 선택",
            ["📹 실시간 카메라", "📁 비디오 파일 업로드"],
            index=1,  # 기본값을 파일 업로드로 변경
        )

        st.markdown("---")

        # 추적 설정
        st.subheader("🎯 추적 설정")
        confidence_threshold = st.slider("신뢰도 임계값", 0.1, 1.0, 0.6, 0.1)

        # 트래커 초기화 버튼
        if st.button("🔧 트래커 초기화"):
            app.initialize_tracker(confidence_threshold)

        st.markdown("---")

        # 비디오 소스별 설정
        if video_source_type == "📹 실시간 카메라":
            st.subheader("📹 카메라 설정")
            camera_id = st.number_input("카메라 ID", min_value=0, max_value=10, value=0)

            st.info("💡 macOS에서는 터미널에 카메라 권한이 필요할 수 있습니다.")

            if st.button("📹 카메라 연결"):
                app.process_camera(camera_id)

        else:  # 비디오 파일 업로드
            st.subheader("📁 파일 업로드")
            uploaded_file = st.file_uploader(
                "비디오 파일 선택",
                type=["mp4", "avi", "mov", "mkv"],
                help="MP4, AVI, MOV, MKV 형식을 지원합니다.",
            )

            if uploaded_file is not None:
                if st.button("📁 파일 로드"):
                    app.process_video_file(uploaded_file, confidence_threshold)

        st.markdown("---")

        # 제어 버튼
        st.subheader("🎮 제어")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ 시작", use_container_width=True):
                if app.video_source and app.tracker:
                    app.is_running = True
                    st.success("추적을 시작합니다!")
                else:
                    st.error("비디오 소스와 트래커를 먼저 설정해주세요.")

        with col2:
            if st.button("⏹️ 정지", use_container_width=True):
                app.is_running = False
                st.info("추적이 정지되었습니다.")

        if st.button("🔄 리셋", use_container_width=True):
            app.cleanup()
            st.session_state.app = StreamlitApp()
            st.success("시스템이 리셋되었습니다.")

    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 실시간 모니터링")
        video_placeholder = st.empty()

        # 기본 이미지 표시
        if not app.is_running:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                dummy_frame,
                "Video Stream Ready...",
                (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            video_placeholder.image(
                dummy_frame, channels="BGR", use_container_width=True
            )

    with col2:
        st.subheader("📊 실시간 통계")
        stats_placeholder = st.empty()

        # 기본 통계 표시
        with stats_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("프레임", app.frame_count)
            with col2:
                st.metric("총 감지", app.detection_count)
            with col3:
                st.metric("라인 교차", app.crossing_count)

        st.markdown("---")

        st.subheader("🔔 최근 이벤트")
        events_placeholder = st.empty()

        with events_placeholder.container():
            st.info("추적을 시작하면 이벤트가 표시됩니다.")

    # 추적 실행
    if app.is_running:
        app.run_tracking(video_placeholder, stats_placeholder, events_placeholder)


if __name__ == "__main__":
    main()
