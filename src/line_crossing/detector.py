"""
라인 크로싱 감지기

객체가 가상 라인을 교차할 때 이벤트를 생성하는 모듈
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from ..core.enums import CrossingDirection
from ..core.models import CrossingEvent, DetectionResult, VirtualLine
from .manager import LineManager

logger = logging.getLogger(__name__)


class TrackHistory:
    """객체 추적 이력 관리"""

    def __init__(self, track_id: int, max_history: int = 10):
        self.track_id = track_id
        self.positions = []  # (x, y, timestamp) 튜플들
        self.max_history = max_history
        self.last_crossing_time = {}  # line_id -> timestamp

    def add_position(self, x: float, y: float, timestamp: float):
        """새 위치 추가"""
        self.positions.append((x, y, timestamp))

        # 최대 이력 수 제한
        if len(self.positions) > self.max_history:
            self.positions = self.positions[-self.max_history :]

    def get_recent_positions(self, count: int = 2) -> List[Tuple[float, float, float]]:
        """최근 위치들 반환"""
        return self.positions[-count:] if len(self.positions) >= count else []

    def can_generate_crossing_event(
        self, line_id: str, min_interval: float = 1.0
    ) -> bool:
        """교차 이벤트 생성 가능 여부 (중복 방지)"""
        last_time = self.last_crossing_time.get(line_id, 0)
        current_time = time.time()
        return (current_time - last_time) >= min_interval

    def record_crossing(self, line_id: str):
        """교차 이벤트 기록"""
        self.last_crossing_time[line_id] = time.time()


class LineCrossingDetector:
    """라인 크로싱 감지기"""

    def __init__(self, line_manager: LineManager, crossing_threshold: float = 5.0):
        """
        Args:
            line_manager: 라인 관리자
            crossing_threshold: 교차 감지 임계값 (픽셀)
        """
        self.line_manager = line_manager
        self.crossing_threshold = crossing_threshold
        self.track_histories: Dict[int, TrackHistory] = {}
        self.min_crossing_interval = 1.0  # 초 단위

        logger.info(f"라인 크로싱 감지기 초기화 (임계값: {crossing_threshold}px)")

    def detect_crossing(self, detection: DetectionResult) -> List[CrossingEvent]:
        """객체의 라인 교차 감지"""
        crossing_events = []

        try:
            track_id = detection.track_id
            current_pos = detection.center_point
            timestamp = detection.timestamp

            # 추적 이력 업데이트
            if track_id not in self.track_histories:
                self.track_histories[track_id] = TrackHistory(track_id)

            history = self.track_histories[track_id]
            history.add_position(current_pos[0], current_pos[1], timestamp)

            # 최근 위치가 충분하지 않으면 교차 감지 불가
            recent_positions = history.get_recent_positions(2)
            if len(recent_positions) < 2:
                return crossing_events

            # 활성 라인들에 대해 교차 검사
            active_lines = self.line_manager.get_active_lines()
            for line_id, line in active_lines.items():

                # 중복 이벤트 방지
                if not history.can_generate_crossing_event(
                    line_id, self.min_crossing_interval
                ):
                    continue

                # 교차 검사
                crossing_result = self._check_line_crossing(
                    recent_positions, line, detection
                )

                if crossing_result:
                    crossing_point, direction = crossing_result

                    # 교차 이벤트 생성
                    event = CrossingEvent(
                        event_id=str(uuid.uuid4()),
                        track_id=track_id,
                        line_id=line_id,
                        direction=direction,
                        crossing_point=crossing_point,
                        timestamp=timestamp,
                        confidence=detection.confidence,
                        detection_result=detection,
                    )

                    crossing_events.append(event)
                    history.record_crossing(line_id)

                    logger.info(
                        f"라인 교차 감지: ID {track_id}, 라인 {line_id}, "
                        f"방향 {direction.value}, 위치 {crossing_point}"
                    )

        except Exception as e:
            logger.error(f"라인 교차 감지 중 오류: {e}")

        return crossing_events

    def _check_line_crossing(
        self,
        positions: List[Tuple[float, float, float]],
        line: VirtualLine,
        detection: DetectionResult,
    ) -> Optional[Tuple[Tuple[float, float], CrossingDirection]]:
        """라인 교차 여부 확인"""

        if len(positions) < 2:
            return None

        # 이전 위치와 현재 위치
        prev_pos = (positions[0][0], positions[0][1])
        curr_pos = (positions[1][0], positions[1][1])

        # 움직임 경로 생성
        movement_line = LineString([prev_pos, curr_pos])

        # 가상 라인과 교차점 확인
        if movement_line.intersects(line.geometry):
            intersection = movement_line.intersection(line.geometry)

            # 교차점 좌표 추출
            if hasattr(intersection, "x") and hasattr(intersection, "y"):
                crossing_point = (intersection.x, intersection.y)
            else:
                # 복수 교차점인 경우 첫 번째 점 사용
                crossing_point = (curr_pos[0], curr_pos[1])

            # 교차 방향 결정
            direction = self._determine_crossing_direction(prev_pos, curr_pos, line)

            return crossing_point, direction

        return None

    def _determine_crossing_direction(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        line: VirtualLine,
    ) -> CrossingDirection:
        """교차 방향 결정"""

        try:
            # 라인의 방향 벡터
            line_start = line.start_point
            line_end = line.end_point
            line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])

            # 객체 이동 벡터
            movement_vector = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])

            # 외적을 이용한 방향 판단
            cross_product = (
                line_vector[0] * movement_vector[1]
                - line_vector[1] * movement_vector[0]
            )

            # 외적의 부호로 방향 결정
            if cross_product > 0:
                return CrossingDirection.IN
            elif cross_product < 0:
                return CrossingDirection.OUT
            else:
                return CrossingDirection.UNKNOWN

        except Exception as e:
            logger.warning(f"교차 방향 결정 실패: {e}")
            return CrossingDirection.UNKNOWN

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """비활성 추적 이력 정리"""
        # 현재 활성화된 추적 ID가 아닌 이력들 제거
        inactive_ids = [
            track_id
            for track_id in self.track_histories.keys()
            if track_id not in active_track_ids
        ]

        for track_id in inactive_ids:
            del self.track_histories[track_id]

        if inactive_ids:
            logger.debug(f"비활성 추적 이력 {len(inactive_ids)}개 정리 완료")

    def get_statistics(self) -> Dict[str, int]:
        """통계 정보 반환"""
        return {
            "active_tracks": len(self.track_histories),
            "active_lines": len(self.line_manager.get_active_lines()),
            "total_lines": self.line_manager.get_line_count(),
        }

    def reset(self):
        """감지기 초기화"""
        self.track_histories.clear()
        logger.info("라인 크로싱 감지기 초기화 완료")
