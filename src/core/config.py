"""
하드코딩된 기본 설정값

MVP에서는 복잡한 설정 관리 대신 기본값을 하드코딩합니다.
라인 설정만 line_configs.json으로 동적 관리합니다.
"""

# 추적 관련 기본 설정
DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # 객체 감지 신뢰도 임계값
DEFAULT_MAX_AGE = 30  # 추적 객체 최대 유지 프레임 수
DEFAULT_MIN_HITS = 3  # 추적 시작을 위한 최소 감지 횟수
DEFAULT_IOU_THRESHOLD = 0.3  # IoU 임계값

# 비디오 처리 설정
DEFAULT_FRAME_WIDTH = 640  # 기본 프레임 너비
DEFAULT_FRAME_HEIGHT = 480  # 기본 프레임 높이
DEFAULT_FPS = 30  # 기본 FPS

# 라인 교차 감지 설정
DEFAULT_CROSSING_THRESHOLD = 0.5  # 라인 교차 감지 임계값
DEFAULT_COOLDOWN_FRAMES = 30  # 중복 감지 방지 쿨다운 프레임 수

# 라인 시각화 설정
DEFAULT_LINE_THICKNESS = 3  # 기본 라인 두께
DEFAULT_LINE_COLOR = (0, 255, 0)  # 기본 라인 색상 (BGR)
DEFAULT_ACTIVE_LINE_COLOR = (0, 255, 0)  # 활성 라인 색상
DEFAULT_INACTIVE_LINE_COLOR = (128, 128, 128)  # 비활성 라인 색상

# 스냅샷 설정
SNAPSHOT_QUALITY = 95  # JPEG 품질 (1-100)
SNAPSHOT_MAX_WIDTH = 800  # 스냅샷 최대 너비
SNAPSHOT_MAX_HEIGHT = 600  # 스냅샷 최대 높이

# 파일 경로 설정
LINE_CONFIG_FILE = "configs/line_configs.json"
SNAPSHOT_DIR = "data/snapshots"
LOG_DIR = "data/logs"
TEMP_DIR = "data/temp"

# 로그 설정
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CSV_DATE_FORMAT = "%Y-%m-%d"
