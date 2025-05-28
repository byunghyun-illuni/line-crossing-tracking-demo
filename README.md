# Line Crossing Tracking Demo

MMTracking + OC-SORT 기반 2D 영상에서 가상 라인을 통한 실시간 출입 감지 및 모니터링 시스템

## 🎯 프로젝트 개요

이 프로젝트는 MMTracking과 OC-SORT 알고리즘을 활용하여 2D 영상에서 가상 라인을 설정하고, 객체가 해당 라인을 교차할 때를 실시간으로 감지하는 시스템입니다. Streamlit을 통한 웹 인터페이스를 제공하여 사용자가 쉽게 라인을 설정하고 모니터링할 수 있습니다.

## 🚀 주요 기능

- **실시간 객체 추적**: MMTracking + OC-SORT 기반 다중 객체 추적
- **가상 라인 관리**: 웹 인터페이스를 통한 라인 생성/수정/삭제
- **교차 감지**: 실시간 라인 교차 이벤트 감지 및 로깅
- **다양한 입력 지원**: MP4 파일 및 실시간 카메라 입력
- **모니터링 대시보드**: Streamlit 기반 실시간 모니터링

## 📋 요구사항

- Python 3.11 또는 3.12
- Windows 10/11 (현재 테스트 환경)
- 웹캠 또는 MP4 비디오 파일

## 🛠️ 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd line-crossing-tracking-demo
```

### 2. 가상환경 생성 및 활성화
```bash
# uv 사용 (권장)
uv venv --python 3.11
.venv\Scripts\activate  # Windows

# 또는 conda 사용
conda create -n line-crossing python=3.11
conda activate line-crossing
```

### 3. 패키지 설치
```bash
# uv 사용 (권장)
uv pip install -e .

# 또는 pip 사용
pip install -e .
```

## 📦 설치된 주요 패키지

- **torch**: 2.7.0+ (딥러닝 프레임워크)
- **torchvision**: 0.22.0+ (컴퓨터 비전)
- **opencv-python**: 4.11.0+ (영상 처리)
- **streamlit**: 1.45.1+ (웹 인터페이스)
- **mmcv**: 2.2.0+ (OpenMMLab 기반 라이브러리)
- **mmdet**: 3.3.0+ (객체 검출)
- **shapely**: 2.1.1+ (기하학적 연산)

## 🎮 사용 방법

### 기본 테스트
```bash
# 패키지 import 테스트
python -c "import torch; import cv2; import streamlit; import shapely; print('설치 완료!')"
```

### Streamlit 앱 실행 (개발 예정)
```bash
streamlit run streamlit_app/main.py
```

## 📁 프로젝트 구조

```
line-crossing-tracking-demo/
├── src/                          # 핵심 비즈니스 로직
│   ├── core/                     # 데이터 모델
│   ├── tracking/                 # MMTracking + OC-SORT
│   ├── line_crossing/            # 라인 교차 감지
│   ├── video/                    # 비디오 처리
│   ├── events/                   # 이벤트 관리
│   └── utils/                    # 유틸리티
├── streamlit_app/                # Streamlit 웹 앱
├── configs/                      # 설정 파일
├── data/                         # 데이터 저장소
├── pyproject.toml               # 프로젝트 설정
└── README.md
```

## 🔧 개발 상태

- ✅ 기본 환경 설정 및 패키지 설치
- ✅ pyproject.toml 구성
- 🚧 MMTracking 통합 (진행 중)
- 🚧 OC-SORT 구현 (예정)
- 🚧 Streamlit 인터페이스 (예정)
- 🚧 라인 교차 감지 로직 (예정)

## 📝 참고사항

- MMTracking 설치 시 일부 종속성 문제가 있을 수 있습니다. 이는 향후 해결 예정입니다.
- 현재는 기본 패키지들만 설치되어 있으며, 실제 추적 기능은 개발 중입니다.
- GPU 사용을 위해서는 CUDA 버전의 PyTorch 설치가 필요할 수 있습니다.

## 🤝 기여

이 프로젝트는 MVP(Minimum Viable Product) 단계입니다. 기여나 제안사항이 있으시면 이슈를 생성해 주세요.

## 📄 라이선스

Apache-2.0 License
