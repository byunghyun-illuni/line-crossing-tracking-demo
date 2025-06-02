"""
라인 크로싱 감지기 - 완전 리팩토링 버전

객체가 가상 라인을 교차할 때 이벤트를 생성하는 단순하고 효과적인 모듈
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VirtualLine:
    """가상 라인 클래스 (단순화된 버전)"""

    def __init__(
        self,
        line_id: str,
        name: str,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
        is_active: bool = True,
    ):
        self.line_id = line_id
        self.name = name
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.thickness = thickness
        self.is_active = is_active


class TrackHistory:
    """객체 추적 이력 관리 (단순화된 버전)"""

    def __init__(self, track_id: int, max_history: int = 5):
        self.track_id = track_id
        self.positions = []  # (x, y) 튜플들
        self.max_history = max_history
        self.last_crossing_time = {}  # line_id -> timestamp

    def add_position(self, x: float, y: float):
        """새 위치 추가"""
        self.positions.append((x, y))

        # 최대 이력 수 제한
        if len(self.positions) > self.max_history:
            self.positions = self.positions[-self.max_history :]

    def get_recent_positions(self, count: int = 2) -> List[Tuple[float, float]]:
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
    """라인 크로싱 감지기 (완전 리팩토링 버전)"""

    def __init__(self):
        self.track_histories: Dict[int, TrackHistory] = {}
        self.crossing_stats = {
            "total_in": 0,
            "total_out": 0,
            "line_stats": {},  # line_id -> {'in': count, 'out': count}
        }
        logger.info("라인 크로싱 감지기 초기화 완료")

    def detect_crossings(
        self, detections: List, lines: Dict[str, VirtualLine]
    ) -> List[str]:
        """라인 교차 감지 - 메인 로직"""
        crossing_events = []

        for detection in detections:
            track_id = detection.track_id
            if track_id <= 0:  # 유효한 track_id가 아니면 무시
                continue

            center_x, center_y = detection.center_point

            # 추적 이력 업데이트
            if track_id not in self.track_histories:
                self.track_histories[track_id] = TrackHistory(track_id)

            history = self.track_histories[track_id]
            history.add_position(center_x, center_y)

            # 최근 위치가 충분하지 않으면 교차 감지 불가
            recent_positions = history.get_recent_positions(2)
            if len(recent_positions) < 2:
                continue

            # 각 라인에 대해 교차 검사
            for line_id, line in lines.items():
                if not line.is_active:
                    continue

                # 중복 이벤트 방지
                if not history.can_generate_crossing_event(line_id, 1.0):
                    continue

                # 교차 검사
                crossing_direction = self._check_line_crossing(recent_positions, line)

                if crossing_direction:
                    # 교차 이벤트 기록
                    history.record_crossing(line_id)

                    # 통계 업데이트
                    if line_id not in self.crossing_stats["line_stats"]:
                        self.crossing_stats["line_stats"][line_id] = {"in": 0, "out": 0}

                    if crossing_direction == "IN":
                        self.crossing_stats["total_in"] += 1
                        self.crossing_stats["line_stats"][line_id]["in"] += 1
                    else:
                        self.crossing_stats["total_out"] += 1
                        self.crossing_stats["line_stats"][line_id]["out"] += 1

                    event_msg = f"ID {track_id}: {crossing_direction} ({line.name})"
                    crossing_events.append(event_msg)
                    logger.info(f"라인 교차 감지: {event_msg}")

        return crossing_events

    def _check_line_crossing(
        self, positions: List[Tuple[float, float]], line: VirtualLine
    ) -> Optional[str]:
        """라인 교차 여부 및 방향 확인"""
        if len(positions) < 2:
            return None

        prev_pos = positions[0]
        curr_pos = positions[1]

        # 라인의 시작점과 끝점
        line_start = line.start_point
        line_end = line.end_point

        # 라인과 이동경로의 교차점 확인
        if self._lines_intersect(prev_pos, curr_pos, line_start, line_end):
            # 교차 방향 결정
            direction = self._determine_crossing_direction(prev_pos, curr_pos, line)
            return direction

        return None

    def _lines_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> bool:
        """두 선분이 교차하는지 확인 - CCW 알고리즘 사용"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _determine_crossing_direction(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        line: VirtualLine,
    ) -> str:
        """교차 방향 결정 - 외적을 이용한 방법"""
        try:
            # 라인의 방향 벡터
            line_start = line.start_point
            line_end = line.end_point
            line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])

            # 객체 이동 벡터
            movement_vector = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])

            # 외적을 이용한 방향 판단 (cross product)
            cross_product = (
                line_vector[0] * movement_vector[1]
                - line_vector[1] * movement_vector[0]
            )

            # 외적의 부호로 방향 결정
            if cross_product > 0:
                return "OUT"
            elif cross_product < 0:
                return "IN"
            else:
                return "UNKNOWN"

        except Exception as e:
            logger.warning(f"교차 방향 결정 실패: {e}")
            return "UNKNOWN"

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """비활성 추적 이력 정리"""
        inactive_ids = [
            track_id
            for track_id in self.track_histories.keys()
            if track_id not in active_track_ids
        ]

        for track_id in inactive_ids:
            del self.track_histories[track_id]

        if inactive_ids:
            logger.debug(f"비활성 추적 이력 {len(inactive_ids)}개 정리 완료")

    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        return {
            "total_in": self.crossing_stats["total_in"],
            "total_out": self.crossing_stats["total_out"],
            "net_count": self.crossing_stats["total_in"]
            - self.crossing_stats["total_out"],
            "line_stats": self.crossing_stats["line_stats"].copy(),
            "active_tracks": len(self.track_histories),
        }

    def reset(self):
        """감지기 초기화"""
        self.track_histories.clear()
        self.crossing_stats = {"total_in": 0, "total_out": 0, "line_stats": {}}
        logger.info("라인 크로싱 감지기 초기화 완료")
