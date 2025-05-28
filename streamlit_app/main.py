"""
2D Access Control MVP - Streamlit 메인 애플리케이션

MMTracking + OC-SORT 기반 라인 크로싱 추적 시스템
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 설정
st.set_page_config(
    page_title="2D Access Control System",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """메인 애플리케이션"""
    st.title("🚪 2D Access Control MVP")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 비디오 소스 선택
        video_source = st.selectbox(
            "비디오 소스",
            ["웹캠", "MP4 파일"],
            index=0
        )
        
        if video_source == "웹캠":
            camera_id = st.number_input("카메라 ID", min_value=0, max_value=10, value=0)
        else:
            uploaded_file = st.file_uploader("MP4 파일 업로드", type=['mp4', 'avi', 'mov'])
        
        st.markdown("---")
        
        # 추적 설정
        st.subheader("🎯 추적 설정")
        confidence_threshold = st.slider("신뢰도 임계값", 0.1, 1.0, 0.6, 0.1)
        
        st.markdown("---")
        
        # 라인 관리
        st.subheader("📏 가상 라인 관리")
        if st.button("새 라인 추가"):
            st.info("비디오 화면에서 클릭하여 라인을 그려주세요")
        
        if st.button("모든 라인 삭제"):
            st.warning("모든 라인이 삭제됩니다")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 실시간 모니터링")
        
        # 비디오 플레이스홀더
        video_placeholder = st.empty()
        
        # 임시 이미지 (실제 비디오 스트림 대신)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Video Stream Coming Soon...", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        video_placeholder.image(dummy_frame, channels="BGR", use_column_width=True)
        
        # 제어 버튼
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            start_btn = st.button("▶️ 시작", use_container_width=True)
        with col_btn2:
            stop_btn = st.button("⏹️ 정지", use_container_width=True)
        with col_btn3:
            reset_btn = st.button("🔄 리셋", use_container_width=True)
    
    with col2:
        st.subheader("📊 실시간 통계")
        
        # 메트릭 표시
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("총 감지", "0", "0")
        with col_metric2:
            st.metric("라인 교차", "0", "0")
        
        st.markdown("---")
        
        # 최근 이벤트
        st.subheader("🔔 최근 이벤트")
        
        # 이벤트 리스트 (임시)
        events_placeholder = st.empty()
        with events_placeholder.container():
            st.info("아직 이벤트가 없습니다")
        
        st.markdown("---")
        
        # 활성 라인 목록
        st.subheader("📏 활성 라인")
        lines_placeholder = st.empty()
        with lines_placeholder.container():
            st.info("설정된 라인이 없습니다")

if __name__ == "__main__":
    main() 