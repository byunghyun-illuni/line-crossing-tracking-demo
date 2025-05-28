#!/usr/bin/env python3
"""
Streamlit 앱 실행 스크립트
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """메인 실행 함수"""
    # 프로젝트 루트 디렉토리로 이동
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # macOS 카메라 권한 문제 해결을 위한 환경 변수 설정
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

    print("🚀 2D Access Control MVP 시작 중...")
    print(f"📁 프로젝트 디렉토리: {project_root}")
    print(
        "🎥 테스트 비디오가 필요하면 먼저 'python create_test_video.py'를 실행하세요."
    )
    print("=" * 60)

    try:
        # Streamlit 앱 실행
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app/main.py",
            "--server.port",
            "8501",
            "--server.headless",
            "false",
        ]

        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n👋 애플리케이션이 종료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"❌ 애플리케이션 실행 중 오류 발생: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
