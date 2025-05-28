"""
핵심 데이터 모델

추적, 감지, 교차 이벤트 등의 핵심 데이터 구조를 정의합니다.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from .enums import CameraViewType, CrossingDirection


@dataclass
class DetectionResult:
    """객체 감지 결과"""

    track_id: int  # 추적 ID
    bbox: Tuple[int, int, int, int]  # 바운딩 박스 (x, y, w, h)
    center_point: Tuple[float, float]  # 중심점 (x, y)
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    class_name: str  # 클래스 이름 (예: "person")
    timestamp: float  # 타임스탬프
    features: Dict[str, Any] = field(default_factory=dict)  # 추가 특징

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "center_point": self.center_point,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "timestamp": self.timestamp,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """딕셔너리에서 생성"""
        return cls(
            track_id=data["track_id"],
            bbox=tuple(data["bbox"]),
            center_point=tuple(data["center_point"]),
            confidence=data["confidence"],
            class_name=data["class_name"],
            timestamp=data["timestamp"],
            features=data.get("features", {}),
        )


@dataclass
class TrackingFrame:
    """프레임별 추적 정보"""

    frame_id: int  # 프레임 ID
    timestamp: float  # 타임스탬프
    detections: List[DetectionResult]  # 감지된 객체들
    raw_frame: Optional[np.ndarray] = None  # 원본 프레임 (메모리 절약을 위해 Optional)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 메타데이터

    def get_detection_by_id(self, track_id: int) -> Optional[DetectionResult]:
        """추적 ID로 감지 결과 조회"""
        for detection in self.detections:
            if detection.track_id == track_id:
                return detection
        return None

    def filter_by_confidence(self, threshold: float) -> List[DetectionResult]:
        """신뢰도 임계값으로 필터링"""
        return [d for d in self.detections if d.confidence >= threshold]


@dataclass
class CrossingEvent:
    """라인 교차 이벤트"""

    event_id: str  # 이벤트 ID
    track_id: int  # 추적 ID
    line_id: str  # 라인 ID
    direction: CrossingDirection  # 교차 방향
    crossing_point: Tuple[float, float]  # 교차점 좌표
    timestamp: float  # 타임스탬프
    confidence: float  # 신뢰도
    detection_result: DetectionResult  # 감지 결과
    metadata: Dict[str, Any] = field(default_factory=dict)  # 메타데이터

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        data = {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "line_id": self.line_id,
            "direction": self.direction.value,
            "crossing_point": self.crossing_point,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "detection_result": self.detection_result.to_dict(),
            "metadata": self.metadata,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CrossingEvent":
        """JSON 문자열에서 생성"""
        data = json.loads(json_str)
        return cls(
            event_id=data["event_id"],
            track_id=data["track_id"],
            line_id=data["line_id"],
            direction=CrossingDirection(data["direction"]),
            crossing_point=tuple(data["crossing_point"]),
            timestamp=data["timestamp"],
            confidence=data["confidence"],
            detection_result=DetectionResult.from_dict(data["detection_result"]),
            metadata=data.get("metadata", {}),
        )

    def get_datetime(self) -> datetime:
        """타임스탬프를 datetime으로 변환"""
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class VirtualLine:
    """가상 라인"""

    line_id: str  # 라인 ID
    name: str  # 라인 이름
    start_point: Tuple[float, float]  # 시작점 (x, y)
    end_point: Tuple[float, float]  # 끝점 (x, y)
    is_active: bool = True  # 활성 상태
    view_type: CameraViewType = CameraViewType.ENTRANCE  # 카메라 뷰 타입
    thickness: float = 3.0  # 라인 두께
    color: Tuple[int, int, int] = (0, 255, 0)  # 라인 색상 (BGR)
    direction_config: Dict[str, Any] = field(default_factory=dict)  # 방향 설정

    def __post_init__(self):
        """초기화 후 처리"""
        if not self.direction_config:
            self.direction_config = self._auto_detect_direction_config()

    @property
    def geometry(self) -> LineString:
        """Shapely LineString 객체"""
        return LineString([self.start_point, self.end_point])

    def validate_points(self) -> bool:
        """라인 좌표 유효성 검사"""
        try:
            x1, y1 = self.start_point
            x2, y2 = self.end_point

            # 좌표가 숫자인지 확인
            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                return False

            # 시작점과 끝점이 다른지 확인
            if (x1, y1) == (x2, y2):
                return False

            # 좌표가 양수인지 확인 (화면 좌표계)
            if any(coord < 0 for coord in [x1, y1, x2, y2]):
                return False

            return True
        except (ValueError, TypeError):
            return False

    def calculate_distance_to_point(self, point: Tuple[float, float]) -> float:
        """점과 라인 사이의 거리 계산"""
        shapely_point = Point(point)
        return self.geometry.distance(shapely_point)

    def get_perpendicular_distance(self, point: Tuple[float, float]) -> float:
        """점에서 라인까지의 수직 거리"""
        return self.calculate_distance_to_point(point)

    def _auto_detect_direction_config(self) -> Dict[str, Any]:
        """방향 설정 자동 감지"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # 라인의 각도 계산
        angle = np.arctan2(y2 - y1, x2 - x1)
        angle_degrees = np.degrees(angle)

        # 수직/수평 라인 판단
        is_horizontal = abs(angle_degrees) < 30 or abs(angle_degrees) > 150
        is_vertical = 60 < abs(angle_degrees) < 120

        return {
            "angle": angle_degrees,
            "is_horizontal": is_horizontal,
            "is_vertical": is_vertical,
            "auto_generated": True,
        }

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 저장용)"""
        return {
            "line_id": self.line_id,
            "name": self.name,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "is_active": self.is_active,
            "view_type": self.view_type.value,
            "thickness": self.thickness,
            "color": self.color,
            "direction_config": self.direction_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VirtualLine":
        """딕셔너리에서 생성"""
        return cls(
            line_id=data["line_id"],
            name=data["name"],
            start_point=tuple(data["start_point"]),
            end_point=tuple(data["end_point"]),
            is_active=data.get("is_active", True),
            view_type=CameraViewType(data.get("view_type", "entrance")),
            thickness=data.get("thickness", 3.0),
            color=tuple(data.get("color", [0, 255, 0])),
            direction_config=data.get("direction_config", {}),
        )
