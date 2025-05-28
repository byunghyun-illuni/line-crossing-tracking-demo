"""
2D Access Control MVP - Streamlit 메인 애플리케이션

OC-SORT 기반 라인 크로싱 추적 시스템 (YOLOX 검출기 포함)
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("streamlit_app.log"),
    ],
)
logger = logging.getLogger(__name__)

# macOS 카메라 권한 문제 해결을 위한 환경 변수 설정
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models import DetectionResult
from src.tracking.detector_configs import list_configs
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
        self.video_source = None
        self.tracker = None
        self.is_running = False
        self.frame_count = 0
        self.detection_count = 0
        self.crossing_count = 0
        # 상태 추적을 위한 플래그 추가
        self.video_loaded = False
        self.tracker_initialized = False

    def initialize_tracker(
        self, confidence_threshold: float, detector_config: str = "balanced"
    ):
        """트래커 초기화 (YOLOX 검출기 포함)"""
        try:
            logger.info(
                f"트래커 초기화 중... (검출기: {detector_config}, 신뢰도: {confidence_threshold})"
            )

            self.tracker = ObjectTracker(
                det_thresh=confidence_threshold, detector_config=detector_config
            )
            self.tracker_initialized = True
            logger.info(f"트래커 초기화 완료 (YOLOX {detector_config} 검출기)")
        except Exception as e:
            logger.error(f"트래커 초기화 실패: {e}")
            self.tracker_initialized = False
            raise

    def process_video_file(
        self,
        uploaded_file,
        confidence_threshold: float,
        detector_config: str = "balanced",
    ):
        """비디오 파일 처리"""
        try:
            logger.info(f"비디오 파일 처리 시작: {uploaded_file.name}")
            st.info(f"📁 파일 처리 중: {uploaded_file.name}")

            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                file_content = uploaded_file.read()
                tmp_file.write(file_content)
                temp_path = tmp_file.name

            logger.info(f"임시 파일 생성: {temp_path}, 크기: {len(file_content)} bytes")
            st.info(f"📄 임시 파일 생성 완료: {len(file_content)} bytes")

            # 파일 존재 확인
            if not os.path.exists(temp_path):
                error_msg = f"임시 파일이 생성되지 않았습니다: {temp_path}"
                logger.error(error_msg)
                st.error(error_msg)
                self.video_loaded = False
                return False

            # 비디오 소스 초기화
            logger.info("VideoSource 초기화 중...")
            st.info("🎥 비디오 소스 초기화 중...")

            self.video_source = VideoSource()

            logger.info(f"비디오 파일 열기 시도: {temp_path}")
            st.info("📂 비디오 파일 열기 중...")

            if not self.video_source.open(temp_path):
                error_msg = f"비디오 파일을 열 수 없습니다: {temp_path}"
                logger.error(error_msg)
                st.error(error_msg)
                self.video_loaded = False

                # 임시 파일 정리
                try:
                    os.unlink(temp_path)
                except:
                    pass
                return False

            logger.info("비디오 파일 열기 성공")
            st.success("✅ 비디오 파일 열기 성공!")

            # 트래커 초기화 (YOLOX 검출기 포함)
            logger.info(f"트래커 초기화 중... (검출기: {detector_config})")
            st.info(f"🎯 트래커 초기화 중... (YOLOX {detector_config} 검출기)")

            try:
                self.initialize_tracker(confidence_threshold, detector_config)
                logger.info("트래커 초기화 성공")
                st.success("✅ 트래커 초기화 성공! (YOLOX 검출기 활성화)")
            except Exception as e:
                error_msg = f"트래커 초기화 실패: {e}"
                logger.error(error_msg)
                st.error(error_msg)
                self.video_loaded = False
                self.tracker_initialized = False
                return False

            # 상태 플래그 설정
            self.video_loaded = True

            # 비디오 정보 표시
            fps = self.video_source.get_fps()
            frame_count = self.video_source.get_frame_count()
            width, height = self.video_source.get_frame_size()

            logger.info(
                f"비디오 정보 - FPS: {fps}, 프레임 수: {frame_count}, 크기: {width}x{height}"
            )

            st.success(
                f"✅ 비디오 파일이 로드되었습니다! "
                f"(총 {frame_count}프레임, {fps} FPS, {width}x{height})"
            )

            # 첫 번째 프레임 미리보기
            logger.info("첫 번째 프레임 읽기 시도...")
            success, first_frame = self.video_source.read_frame()
            if success:
                # 프레임 위치를 처음으로 되돌리기
                self.video_source.set_position(0)

                # 미리보기 표시
                st.image(
                    first_frame,
                    channels="BGR",
                    caption="비디오 첫 번째 프레임 미리보기",
                    use_container_width=True,
                )
                logger.info("첫 번째 프레임 미리보기 표시 완료")
            else:
                logger.warning("첫 번째 프레임을 읽을 수 없습니다")
                st.warning("⚠️ 첫 번째 프레임을 읽을 수 없습니다")

            # 임시 파일 정리 (나중에 사용할 수 있도록 일단 보관)
            # os.unlink(temp_path)

            logger.info("비디오 파일 처리 완료")
            return True

        except Exception as e:
            error_msg = f"비디오 파일 처리 중 오류 발생: {e}"
            st.error(error_msg)
            logger.error(error_msg, exc_info=True)
            self.video_loaded = False
            self.tracker_initialized = False
            return False

    def process_camera(
        self,
        camera_id: int,
        confidence_threshold: float = 0.6,
        detector_config: str = "balanced",
    ):
        """카메라 처리"""
        try:
            logger.info(f"카메라 연결 시도: ID {camera_id}")

            self.video_source = VideoSource()
            if not self.video_source.open(camera_id):
                st.error(f"카메라 {camera_id}에 연결할 수 없습니다.")
                logger.error(f"카메라 연결 실패: ID {camera_id}")
                self.video_loaded = False
                return False

            # 트래커 자동 초기화
            self.initialize_tracker(confidence_threshold, detector_config)

            # 상태 플래그 설정
            self.video_loaded = True

            logger.info(f"카메라 연결 성공: ID {camera_id}")
            st.success(f"✅ 카메라 {camera_id}에 연결되었습니다!")

            # 첫 번째 프레임 미리보기
            success, first_frame = self.video_source.read_frame()
            if success:
                st.image(
                    first_frame,
                    channels="BGR",
                    caption="카메라 첫 번째 프레임 미리보기",
                    use_container_width=True,
                )
                logger.info("카메라 첫 번째 프레임 미리보기 표시 완료")

            return True

        except Exception as e:
            st.error(f"카메라 연결 중 오류 발생: {e}")
            logger.error(f"카메라 연결 오류: {e}", exc_info=True)
            self.video_loaded = False
            self.tracker_initialized = False
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
        if not self.is_ready():
            st.error("❌ 비디오 소스와 트래커를 먼저 초기화해주세요.")
            logger.error("비디오 소스 또는 트래커가 초기화되지 않음")
            return

        logger.info("트래킹 시작")

        # 프레임 처리 루프
        frame_container = st.container()

        while self.is_running:
            success, frame = self.video_source.read_frame()

            if not success:
                logger.warning("비디오 스트림 종료")
                st.warning("📹 비디오 스트림이 종료되었습니다.")
                self.is_running = False
                break

            logger.debug(f"프레임 {self.frame_count} 처리 중")

            # 트래킹 수행
            tracking_frame = self.tracker.track_frame(frame)
            logger.debug(f"감지된 객체 수: {len(tracking_frame.detections)}")

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
                (0, 255, 0),  # 녹색으로 변경
                2,
            )

            # 비디오 표시
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
            with events_placeholder.container():
                if tracking_frame.detections:
                    st.write("**최근 감지된 객체들:**")
                    for det in tracking_frame.detections[-5:]:  # 최근 5개만 표시
                        st.write(
                            f"- ID {det.track_id}: {det.class_name} (신뢰도: {det.confidence:.2f})"
                        )
                        logger.info(
                            f"객체 감지: ID {det.track_id}, 클래스: {det.class_name}, 신뢰도: {det.confidence:.2f}"
                        )
                else:
                    st.write("감지된 객체가 없습니다.")

            # 프레임 레이트 조절 (비디오 파일의 경우)
            if hasattr(self.video_source, "get_fps"):
                fps = self.video_source.get_fps()
                if fps > 0:
                    time.sleep(1.0 / fps)
                else:
                    time.sleep(0.03)  # 기본 30 FPS
            else:
                time.sleep(0.03)

            # Streamlit 업데이트를 위한 짧은 대기
            time.sleep(0.01)

        logger.info("트래킹 종료")

    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        self.video_loaded = False
        self.tracker_initialized = False
        if self.video_source:
            self.video_source.release()
            self.video_source = None
        self.tracker = None

    def is_ready(self):
        """시스템이 준비되었는지 확인"""
        return (
            self.video_loaded
            and self.tracker_initialized
            and self.video_source is not None
            and self.tracker is not None
        )


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

        # 추적 설정
        st.subheader("🎯 추적 설정")
        confidence_threshold = st.slider("신뢰도 임계값", 0.1, 1.0, 0.6, 0.1)
        detector_config = st.selectbox("검출기 설정", list_configs())

        st.markdown("---")

        # 비디오 소스 선택
        video_source_type = st.selectbox(
            "비디오 소스 선택",
            ["📁 비디오 파일 업로드", "📹 실시간 카메라"],
            index=0,  # 기본값을 파일 업로드로 설정
        )

        # 비디오 소스별 설정
        if video_source_type == "📹 실시간 카메라":
            st.subheader("📹 카메라 설정")
            camera_id = st.number_input("카메라 ID", min_value=0, max_value=10, value=0)

            st.info("💡 macOS에서는 터미널에 카메라 권한이 필요할 수 있습니다.")

            if st.button("📹 카메라 연결 및 시작", use_container_width=True):
                with st.spinner("카메라 연결 중..."):
                    success = app.process_camera(
                        camera_id, confidence_threshold, detector_config
                    )
                    if success:
                        app.is_running = True
                        st.success("✅ 카메라 연결 및 추적을 시작합니다!")
                        st.rerun()
                    else:
                        st.error("❌ 카메라 연결에 실패했습니다.")

        else:  # 비디오 파일 업로드
            st.subheader("📁 파일 업로드")
            uploaded_file = st.file_uploader(
                "비디오 파일 선택",
                type=["mp4", "avi", "mov", "mkv"],
                help="MP4, AVI, MOV, MKV 형식을 지원합니다.",
            )

            if uploaded_file is not None:
                if st.button("📁 파일 로드 및 시작", use_container_width=True):
                    with st.spinner("비디오 파일 로드 중..."):
                        success = app.process_video_file(
                            uploaded_file, confidence_threshold, detector_config
                        )
                        if success:
                            app.is_running = True
                            st.success("✅ 비디오 로드 및 추적을 시작합니다!")
                            st.rerun()
                        else:
                            st.error("❌ 비디오 파일 로드에 실패했습니다.")

        st.markdown("---")

        # 제어 버튼
        st.subheader("🎮 제어")

        # 상태 표시 - 개선된 상태 체크
        if app.is_ready():
            st.success("✅ 시스템 준비 완료")

            # 상세 상태 정보
            with st.expander("📋 시스템 상태 상세"):
                st.write(
                    f"🎥 비디오 소스: {'✅ 연결됨' if app.video_loaded else '❌ 연결 안됨'}"
                )
                st.write(
                    f"🎯 트래커: {'✅ 초기화됨' if app.tracker_initialized else '❌ 초기화 안됨'}"
                )
                st.write(
                    f"▶️ 실행 상태: {'🟢 실행 중' if app.is_running else '🟡 대기 중'}"
                )
        else:
            st.warning("⚠️ 비디오 소스를 선택해주세요")

            # 디버그 정보 (개발용)
            if st.checkbox("🔧 디버그 정보 표시"):
                st.write(f"video_loaded: {app.video_loaded}")
                st.write(f"tracker_initialized: {app.tracker_initialized}")
                st.write(f"video_source: {app.video_source is not None}")
                st.write(f"tracker: {app.tracker is not None}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "⏸️ 일시정지" if app.is_running else "▶️ 재생",
                use_container_width=True,
                disabled=not app.is_ready(),
            ):
                if app.is_ready():
                    app.is_running = not app.is_running
                    if app.is_running:
                        st.success("▶️ 추적을 재개합니다!")
                    else:
                        st.info("⏸️ 추적이 일시정지되었습니다.")
                else:
                    st.error("❌ 먼저 비디오를 로드해주세요.")

        with col2:
            if st.button(
                "⏹️ 정지", use_container_width=True, disabled=not app.is_ready()
            ):
                app.is_running = False
                st.info("⏹️ 추적이 정지되었습니다.")

        if st.button("🔄 리셋", use_container_width=True):
            app.cleanup()
            st.session_state.app = StreamlitApp()
            st.success("🔄 시스템이 리셋되었습니다.")
            st.rerun()

    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 실시간 모니터링")
        video_placeholder = st.empty()

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

        st.subheader("📋 이벤트 로그")
        events_placeholder = st.empty()

    # 추적 실행
    if app.is_running and app.is_ready():
        logger.info("추적 시작됨")
        app.run_tracking(video_placeholder, stats_placeholder, events_placeholder)
    else:
        # 기본 이미지 표시
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if app.is_ready():
            message = "Ready to Start - Click Play Button"
            color = (0, 255, 0)  # 녹색
        else:
            message = "Please Upload Video or Connect Camera"
            color = (255, 255, 0)  # 노란색

        cv2.putText(
            dummy_frame,
            message,
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        video_placeholder.image(dummy_frame, channels="BGR", use_container_width=True)


if __name__ == "__main__":
    main()
