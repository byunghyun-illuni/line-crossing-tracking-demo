"""
핵심 열거형 정의

출입 방향과 카메라 뷰 타입을 정의합니다.
"""

from enum import Enum


class CrossingDirection(Enum):
    """라인 교차 방향"""

    IN = "in"  # 들어옴
    OUT = "out"  # 나감
    UNKNOWN = "unknown"  # 방향 불명


class CameraViewType(Enum):
    """카메라 뷰 타입"""

    ENTRANCE = "entrance"  # 출입구
    CORRIDOR = "corridor"  # 복도
    ROOM = "room"  # 방
    OUTDOOR = "outdoor"  # 실외
