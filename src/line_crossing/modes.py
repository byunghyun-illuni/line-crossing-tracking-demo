"""
라인 크로싱 감지 모드 정의

다양한 추적 포인트 모드를 제공하여 상황에 맞는 라인 크로싱 감지를 지원합니다.
"""

from enum import Enum
from typing import Tuple


class TrackingPointMode(Enum):
    """추적 포인트 모드 열거형"""

    CENTER = "center"  # 바운딩 박스 중심점
    BOTTOM_CENTER = "bottom_center"  # 하단 중심점 (발 추적용)
    TOP_CENTER = "top_center"  # 상단 중심점 (머리 추적용)
    BOTTOM_LEFT = "bottom_left"  # 좌하단 모서리
    BOTTOM_RIGHT = "bottom_right"  # 우하단 모서리
    LEFT_CENTER = "left_center"  # 좌측 중심점
    RIGHT_CENTER = "right_center"  # 우측 중심점


def get_tracking_point(detection, mode: TrackingPointMode) -> Tuple[float, float]:
    """
    Detection 객체에서 지정된 모드에 따른 추적 포인트를 계산

    Args:
        detection: Detection 객체 (bbox, center_point 속성 포함)
        mode: 추적 포인트 모드

    Returns:
        (x, y) 좌표 튜플
    """
    x, y, w, h = detection.bbox

    if mode == TrackingPointMode.CENTER:
        # 기존 center_point 사용
        return detection.center_point

    elif mode == TrackingPointMode.BOTTOM_CENTER:
        # 좌하단-우하단 중간점 (발 위치)
        return (x + w / 2, y + h)

    elif mode == TrackingPointMode.TOP_CENTER:
        # 좌상단-우상단 중간점 (머리 위치)
        return (x + w / 2, y)

    elif mode == TrackingPointMode.BOTTOM_LEFT:
        # 좌하단 모서리
        return (x, y + h)

    elif mode == TrackingPointMode.BOTTOM_RIGHT:
        # 우하단 모서리
        return (x + w, y + h)

    elif mode == TrackingPointMode.LEFT_CENTER:
        # 좌측 중심점
        return (x, y + h / 2)

    elif mode == TrackingPointMode.RIGHT_CENTER:
        # 우측 중심점
        return (x + w, y + h / 2)

    else:
        # 기본값: 중심점
        return detection.center_point


def get_mode_description(mode: TrackingPointMode) -> str:
    """모드별 설명 반환"""
    descriptions = {
        TrackingPointMode.CENTER: "바운딩 박스 중심점",
        TrackingPointMode.BOTTOM_CENTER: "하단 중심점 (발 추적용)",
        TrackingPointMode.TOP_CENTER: "상단 중심점 (머리 추적용)",
        TrackingPointMode.BOTTOM_LEFT: "좌하단 모서리",
        TrackingPointMode.BOTTOM_RIGHT: "우하단 모서리",
        TrackingPointMode.LEFT_CENTER: "좌측 중심점",
        TrackingPointMode.RIGHT_CENTER: "우측 중심점",
    }
    return descriptions.get(mode, "알 수 없는 모드")


def get_available_modes() -> list:
    """사용 가능한 모드 리스트 반환"""
    return list(TrackingPointMode)


# 추천 모드별 사용 시나리오
RECOMMENDED_SCENARIOS = {
    TrackingPointMode.CENTER: "일반적인 객체 추적",
    TrackingPointMode.BOTTOM_CENTER: "사람 발걸음 추적 (위에서 아래 시점)",
    TrackingPointMode.TOP_CENTER: "사람 머리 추적 (아래에서 위 시점)",
    TrackingPointMode.BOTTOM_LEFT: "특정 모서리 기준 추적",
    TrackingPointMode.BOTTOM_RIGHT: "특정 모서리 기준 추적",
    TrackingPointMode.LEFT_CENTER: "좌우 이동 감지",
    TrackingPointMode.RIGHT_CENTER: "좌우 이동 감지",
}
