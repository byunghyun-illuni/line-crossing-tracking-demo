#!/usr/bin/env python3
"""
라인 그리기 도구 실행 스크립트

Usage:
    python draw_line.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.line_drawer.manager import LineDrawer


def main():
    """메인 함수"""
    print("=" * 50)
    print("Line Drawing Tool")
    print("=" * 50)

    # 라인 그리기 도구 실행
    drawer = LineDrawer()
    drawer.run()


if __name__ == "__main__":
    main()
