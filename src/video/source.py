"""
비디오 소스 처리

MP4 파일과 실시간 카메라 입력을 통합 처리합니다.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ..core.config import DEFAULT_FPS, DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH

logger = logging.getLogger(__name__)


class VideoSource:
    """비디오 소스 통합 처리 클래스"""

    def __init__(self):
        self.source_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera: bool = False
        self.frame_width: int = DEFAULT_FRAME_WIDTH
        self.frame_height: int = DEFAULT_FRAME_HEIGHT
        self.fps: float = DEFAULT_FPS
        self.is_opened: bool = False

    def open(self, source: Union[str, int]) -> bool:
        """
        비디오 소스 열기

        Args:
            source: 파일 경로(str) 또는 카메라 ID(int)

        Returns:
            bool: 성공 여부
        """
        try:
            # 기존 연결 해제
            self.release()

            # 소스 타입 판단
            if isinstance(source, int):
                # 카메라 입력
                self.is_camera = True
                self.source_path = f"camera_{source}"
                logger.info(f"카메라 {source} 연결 시도")
            else:
                # 파일 입력
                self.is_camera = False
                self.source_path = str(source)

                # 파일 존재 확인
                if not self._validate_source(source):
                    logger.error(f"비디오 파일을 찾을 수 없습니다: {source}")
                    return False

                logger.info(f"비디오 파일 열기: {source}")

            # VideoCapture 객체 생성
            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                logger.error(f"비디오 소스를 열 수 없습니다: {source}")
                return False

            # 비디오 속성 읽기
            self._read_video_properties()

            self.is_opened = True
            logger.info(
                f"비디오 소스 열기 성공: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
            )

            return True

        except Exception as e:
            logger.error(f"비디오 소스 열기 실패: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        프레임 읽기

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (성공 여부, 프레임)
        """
        if not self.is_valid():
            return False, None

        try:
            ret, frame = self.cap.read()

            if not ret:
                if self.is_camera:
                    logger.warning("카메라에서 프레임을 읽을 수 없습니다")
                else:
                    logger.info("비디오 파일 끝에 도달했습니다")
                return False, None

            return True, frame

        except Exception as e:
            logger.error(f"프레임 읽기 실패: {e}")
            return False, None

    def get_fps(self) -> float:
        """FPS 반환"""
        return self.fps

    def get_frame_count(self) -> int:
        """총 프레임 수 반환 (카메라의 경우 -1)"""
        if not self.is_valid() or self.is_camera:
            return -1

        try:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            return -1

    def get_current_position(self) -> int:
        """현재 프레임 위치 반환"""
        if not self.is_valid():
            return -1

        try:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        except:
            return -1

    def set_position(self, frame_number: int) -> bool:
        """
        프레임 위치 설정 (파일만 지원)

        Args:
            frame_number: 이동할 프레임 번호

        Returns:
            bool: 성공 여부
        """
        if not self.is_valid() or self.is_camera:
            return False

        try:
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        except:
            return False

    def get_frame_size(self) -> Tuple[int, int]:
        """프레임 크기 반환 (width, height)"""
        return self.frame_width, self.frame_height

    def release(self) -> None:
        """리소스 해제"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_opened = False
        logger.info("비디오 소스 해제됨")

    def is_valid(self) -> bool:
        """비디오 소스 유효성 확인"""
        return self.is_opened and self.cap is not None and self.cap.isOpened()

    def _validate_source(self, source: str) -> bool:
        """소스 파일 유효성 검사"""
        try:
            path = Path(source)
            return path.exists() and path.is_file()
        except:
            return False

    def _read_video_properties(self) -> None:
        """비디오 속성 읽기"""
        if not self.cap:
            return

        try:
            # 프레임 크기
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps > 0 else DEFAULT_FPS

            # 기본값 설정 (값이 0인 경우)
            if self.frame_width <= 0:
                self.frame_width = DEFAULT_FRAME_WIDTH
            if self.frame_height <= 0:
                self.frame_height = DEFAULT_FRAME_HEIGHT

        except Exception as e:
            logger.warning(f"비디오 속성 읽기 실패, 기본값 사용: {e}")
            self.frame_width = DEFAULT_FRAME_WIDTH
            self.frame_height = DEFAULT_FRAME_HEIGHT
            self.fps = DEFAULT_FPS

    def __enter__(self):
        """컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.release()

    def __del__(self):
        """소멸자"""
        self.release()
