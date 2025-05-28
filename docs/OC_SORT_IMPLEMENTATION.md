# OC-SORT 추적 시스템 구현

## 개요

이 프로젝트는 **공식 OC-SORT (Observation-Centric SORT) 구현**을 사용하여 실시간 객체 추적을 수행합니다. OC-SORT는 기존 SORT 알고리즘을 개선하여 더 정확하고 안정적인 다중 객체 추적을 제공합니다.

## 주요 특징

### 1. 공식 OC-SORT 구현 사용
- **출처**: [noahcao/OC_SORT](https://github.com/noahcao/OC_SORT)
- **논문**: "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking"
- **MMTracking 의존성 없음**: 독립적인 구현으로 Python 3.11/3.12 완전 지원

### 2. 핵심 알고리즘 특징
- **Observation-Centric Re-update**: 관측 중심의 재업데이트 메커니즘
- **Momentum**: 속도 기반 예측 개선
- **Recovery**: 일시적으로 사라진 객체의 복구
- **Velocity Consistency**: 속도 일관성 검사
- **Hungarian Algorithm**: 최적 데이터 연관

### 3. 기술적 구성요소
- **Kalman Filter**: 객체 상태 예측 및 업데이트
- **IoU-based Association**: IoU 기반 데이터 연관
- **Track Lifecycle Management**: 추적 상태 관리 (tentative/confirmed/deleted)

## 아키텍처

```
src/tracking/
├── engine.py                 # ObjectTracker 메인 클래스
├── ocsort_tracker/          # 공식 OC-SORT 구현
│   ├── __init__.py
│   ├── ocsort.py           # 메인 OC-SORT 알고리즘
│   ├── association.py      # 데이터 연관 알고리즘
│   └── utils.py           # 유틸리티 함수들
└── __init__.py
```

## 사용법

### 1. 기본 사용

```python
from src.tracking import ObjectTracker
from src.core.models import DetectionResult
import numpy as np

# 추적기 초기화
tracker = ObjectTracker(
    det_thresh=0.6,      # 감지 임계값
    max_age=30,          # 최대 추적 유지 프레임
    min_hits=3,          # 확정 추적을 위한 최소 히트
    iou_threshold=0.3,   # IoU 임계값
    delta_t=3,           # 속도 계산을 위한 시간 간격
    asso_func="iou",     # 연관 함수
    inertia=0.2,         # 관성 계수
    use_byte=False       # ByteTrack 사용 여부
)

# 프레임 추적
frame = np.zeros((480, 640, 3), dtype=np.uint8)
detections = [...]  # DetectionResult 리스트
tracking_frame = tracker.track_frame(frame, detections)

# 결과 확인
for detection in tracking_frame.detections:
    print(f"Track ID: {detection.track_id}")
    print(f"Bbox: {detection.bbox}")
    print(f"Confidence: {detection.confidence}")
```

### 2. 자동 감지 모드

```python
# 감지 결과 없이 자동 감지 사용 (HOG 기반)
tracking_frame = tracker.track_frame(frame)
```

## 매개변수 설명

### ObjectTracker 매개변수

| 매개변수 | 기본값 | 설명 |
|---------|--------|------|
| `det_thresh` | 0.6 | 감지 신뢰도 임계값 |
| `max_age` | 30 | 추적 유지 최대 프레임 수 |
| `min_hits` | 3 | 확정 추적을 위한 최소 연속 감지 |
| `iou_threshold` | 0.3 | IoU 기반 연관 임계값 |
| `delta_t` | 3 | 속도 계산 시간 간격 |
| `asso_func` | "iou" | 연관 함수 ("iou", "giou", "ciou", "diou", "ct_dist") |
| `inertia` | 0.2 | 속도 예측 관성 계수 |
| `use_byte` | False | ByteTrack 스타일 연관 사용 |

### 매개변수 튜닝 가이드

#### 높은 정확도가 필요한 경우
```python
tracker = ObjectTracker(
    det_thresh=0.7,      # 높은 감지 임계값
    min_hits=5,          # 더 많은 확인 필요
    iou_threshold=0.5,   # 엄격한 연관
)
```

#### 빠른 움직임 대응
```python
tracker = ObjectTracker(
    max_age=50,          # 더 긴 추적 유지
    delta_t=1,           # 짧은 시간 간격
    inertia=0.1,         # 낮은 관성
)
```

#### 혼잡한 환경
```python
tracker = ObjectTracker(
    iou_threshold=0.2,   # 낮은 IoU 임계값
    use_byte=True,       # ByteTrack 연관 사용
    asso_func="giou",    # 더 정교한 연관 함수
)
```

## 데이터 모델

### DetectionResult
```python
@dataclass
class DetectionResult:
    track_id: int                           # 추적 ID
    bbox: Tuple[int, int, int, int]        # (x, y, w, h)
    center_point: Tuple[float, float]      # (center_x, center_y)
    confidence: float                       # 신뢰도 [0.0, 1.0]
    class_name: str                        # 클래스 이름
    timestamp: float                       # 타임스탬프
    features: Dict[str, Any]               # 추가 특징
```

### TrackingFrame
```python
@dataclass
class TrackingFrame:
    frame_id: int                          # 프레임 ID
    timestamp: float                       # 타임스탬프
    detections: List[DetectionResult]      # 추적 결과
    raw_frame: Optional[np.ndarray]        # 원본 프레임 (선택적)
    metadata: Dict[str, Any]               # 메타데이터
```

## 성능 최적화

### 1. 메모리 최적화
- 불필요한 프레임 데이터 제거
- 추적 히스토리 제한
- 효율적인 데이터 구조 사용

### 2. 속도 최적화
- NumPy 벡터화 연산 활용
- 불필요한 계산 최소화
- 적절한 매개변수 설정

### 3. 정확도 최적화
- 적절한 임계값 설정
- 환경에 맞는 연관 함수 선택
- 감지기 품질 개선

## 문제 해결

### 1. 추적 ID 변경 문제
```python
# min_hits 증가로 안정성 향상
tracker = ObjectTracker(min_hits=5)

# IoU 임계값 조정
tracker = ObjectTracker(iou_threshold=0.4)
```

### 2. 빠른 객체 놓침
```python
# max_age 증가
tracker = ObjectTracker(max_age=50)

# delta_t 감소
tracker = ObjectTracker(delta_t=1)
```

### 3. 잘못된 연관
```python
# 더 정교한 연관 함수 사용
tracker = ObjectTracker(asso_func="giou")

# ByteTrack 연관 활성화
tracker = ObjectTracker(use_byte=True)
```

## 테스트

```bash
# 기본 테스트 실행
python test_tracking.py

# 특정 시나리오 테스트
python -c "
from test_tracking import test_ocsort_tracking
test_ocsort_tracking()
"
```

## 의존성

```toml
dependencies = [
    "opencv-python>=4.11.0.86",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "filterpy>=1.4.5",
    "shapely>=2.1.1",
    # ... 기타 의존성
]
```

## 참고 자료

- [OC-SORT 논문](https://arxiv.org/abs/2203.14360)
- [공식 구현](https://github.com/noahcao/OC_SORT)
- [SORT 알고리즘](https://arxiv.org/abs/1602.00763)
- [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)

## 라이선스

이 구현은 공식 OC-SORT 저장소의 라이선스를 따릅니다. 