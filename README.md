# 2D Access Control MVP

**OC-SORT 기반 2D 영상에서 가상 라인을 통한 실시간 출입 감지 및 모니터링 시스템**

## 🎯 프로젝트 개요

이 프로젝트는 OC-SORT (Observation-Centric SORT) 알고리즘을 사용하여 2D 영상에서 객체를 추적하고, 가상 라인을 통한 출입 감지 및 모니터링을 수행하는 MVP 시스템입니다.

## ✨ 주요 기능

- **🎥 다중 비디오 소스 지원**: 실시간 카메라 및 비디오 파일 업로드
- **🎯 고성능 객체 추적**: 공식 OC-SORT 구현체 사용
- **📏 가상 라인 크로싱 감지**: 실시간 출입 모니터링
- **📊 실시간 통계**: 감지 객체 수, 라인 교차 이벤트 추적
- **🖥️ 직관적인 웹 인터페이스**: Streamlit 기반 사용자 친화적 UI

## 🏗️ 시스템 아키텍처

```
src/
├── core/
│   ├── models.py          # 데이터 모델 (DetectionResult, TrackingFrame 등)
│   └── config.py          # 설정 관리
├── tracking/
│   ├── engine.py          # ObjectTracker 메인 클래스
│   └── ocsort_tracker/    # 공식 OC-SORT 구현체
│       ├── ocsort.py      # OC-SORT 메인 알고리즘
│       ├── association.py # 데이터 연관 알고리즘
│       └── kalmanfilter.py # 칼만 필터 구현
├── video/
│   └── source.py          # VideoSource 클래스
├── line_crossing/
│   ├── manager.py         # LineManager 클래스
│   └── detector.py        # CrossingDetector 클래스
└── utils/
    └── visualization.py   # 시각화 유틸리티
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd line-crossing-tracking-demo

# 의존성 설치
pip install -e .
```

### 2. 테스트 비디오 생성

```bash
# 테스트용 비디오 파일 생성
python create_test_video.py
```

### 3. Streamlit 앱 실행

```bash
# 웹 애플리케이션 시작
streamlit run streamlit_app/main.py
```

### 4. 사용 방법

#### 📁 비디오 파일 업로드 방식
1. 사이드바에서 "📁 비디오 파일 업로드" 선택
2. MP4, AVI, MOV, MKV 파일 업로드
3. "📁 파일 로드" 버튼 클릭
4. "🔧 트래커 초기화" 버튼 클릭
5. "▶️ 시작" 버튼으로 추적 시작

#### 📹 실시간 카메라 방식
1. 사이드바에서 "📹 실시간 카메라" 선택
2. 카메라 ID 설정 (기본값: 0)
3. "📹 카메라 연결" 버튼 클릭
4. "🔧 트래커 초기화" 버튼 클릭
5. "▶️ 시작" 버튼으로 추적 시작

## 🔧 기술 스택

### 핵심 알고리즘
- **OC-SORT**: Observation-Centric SORT 추적 알고리즘
- **Kalman Filter**: 객체 상태 예측 및 추적
- **Hungarian Algorithm**: 데이터 연관 최적화
- **HOG Detector**: 기본 객체 감지 (fallback)

### 프레임워크 및 라이브러리
- **Python 3.11+**: 메인 개발 언어
- **OpenCV**: 컴퓨터 비전 처리
- **NumPy**: 수치 연산
- **Streamlit**: 웹 인터페이스
- **FilterPy**: 칼만 필터 구현

## 📊 성능 특징

### OC-SORT 알고리즘 장점
- **관찰 중심 재업데이트**: 가려진 객체의 정확한 추적
- **모멘텀 기반 예측**: 일시적 가림 상황 처리
- **복구 메커니즘**: 손실된 트랙의 자동 복구
- **속도 일관성**: 객체 움직임 패턴 학습

### 실시간 처리
- **30 FPS 지원**: 실시간 비디오 처리
- **낮은 지연시간**: 즉시 감지 및 추적
- **메모리 효율성**: 최적화된 리소스 사용

## 🧪 테스트

### 단위 테스트 실행
```bash
# 추적 시스템 테스트
python test_tracking.py
```

### 테스트 시나리오
- 다중 객체 추적 (3개 객체)
- 객체 가림 및 재등장 처리
- 트랙 ID 일관성 유지
- 바운딩 박스 정확도

## 📈 확장 계획

### 단기 목표
- [ ] 가상 라인 크로싱 감지 완성
- [ ] 실시간 이벤트 알림 시스템
- [ ] 다양한 객체 클래스 지원

### 장기 목표
- [ ] YOLO 기반 고성능 감지기 통합
- [ ] 다중 카메라 지원
- [ ] 클라우드 기반 모니터링 대시보드

## 🤝 기여 방법

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 Apache-2.0 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- [OC-SORT](https://github.com/noahcao/OC_SORT) - 공식 OC-SORT 구현체
- [FilterPy](https://github.com/rlabbe/filterpy) - 칼만 필터 라이브러리
- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리

---

**개발자**: park.byunghyun (byunghyun@illuni.com)  
**버전**: 0.1.0  
**최종 업데이트**: 2024년 12월
