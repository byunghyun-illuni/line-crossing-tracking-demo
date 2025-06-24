"""
단일 가상 라인 관리

단일 가상 라인의 생성, 수정, 저장/로드를 담당합니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.core.config import LINE_CONFIG_FILE
from src.core.enums import CameraViewType
from src.core.models import VirtualLine

logger = logging.getLogger(__name__)


class LineManager:
    """단일 가상 라인 관리 클래스"""

    def __init__(self, config_file_path: str = LINE_CONFIG_FILE):
        self.config_file_path = config_file_path
        self.line: Optional[VirtualLine] = None
        self.auto_save = True

        # 설정 파일 디렉토리 생성
        self._ensure_config_directory()

        # 기존 설정 로드
        self.load_from_file()

    def create_line(
        self, name: str, start_point: tuple, end_point: tuple, **kwargs
    ) -> bool:
        """
        새 라인 생성

        Args:
            name: 라인 이름
            start_point: 시작점 (x, y)
            end_point: 끝점 (x, y)
            **kwargs: 추가 속성 (view_type, thickness, color 등)

        Returns:
            bool: 생성 성공 여부
        """
        try:
            # VirtualLine 객체 생성
            line = VirtualLine(
                line_id="main_line",  # 단일 라인이므로 고정 ID
                name=name,
                start_point=start_point,
                end_point=end_point,
                is_active=kwargs.get("is_active", True),
                view_type=kwargs.get("view_type", CameraViewType.ENTRANCE),
                thickness=kwargs.get("thickness", 4.0),
                color=kwargs.get("color", (0, 0, 255)),
                direction_config=kwargs.get("direction_config", {}),
            )

            # 라인 유효성 검사
            if not line.validate_points():
                logger.error(
                    f"라인 좌표가 유효하지 않습니다: {start_point} -> {end_point}"
                )
                raise ValueError("Invalid line coordinates")

            # 라인 저장
            self.line = line

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            logger.info(f"라인 생성됨: {name}")
            return True

        except Exception as e:
            logger.error(f"라인 생성 실패: {e}")
            return False

    def update_line(self, **kwargs) -> bool:
        """
        라인 업데이트

        Args:
            **kwargs: 업데이트할 속성들

        Returns:
            bool: 성공 여부
        """
        try:
            if self.line is None:
                logger.error("업데이트할 라인이 없습니다")
                return False

            # 속성 업데이트
            for key, value in kwargs.items():
                if hasattr(self.line, key):
                    setattr(self.line, key, value)
                else:
                    logger.warning(f"알 수 없는 속성: {key}")

            # 좌표가 변경된 경우 유효성 검사
            if "start_point" in kwargs or "end_point" in kwargs:
                if not self.line.validate_points():
                    logger.error(f"업데이트된 라인 좌표가 유효하지 않습니다")
                    return False

                # 방향 설정 재계산
                self.line.direction_config = self.line._auto_detect_direction_config()

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            logger.info(f"라인 업데이트됨")
            return True

        except Exception as e:
            logger.error(f"라인 업데이트 실패: {e}")
            return False

    def delete_line(self) -> bool:
        """
        라인 삭제

        Returns:
            bool: 성공 여부
        """
        try:
            if self.line is None:
                logger.error("삭제할 라인이 없습니다")
                return False

            line_name = self.line.name
            self.line = None

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            logger.info(f"라인 삭제됨: {line_name}")
            return True

        except Exception as e:
            logger.error(f"라인 삭제 실패: {e}")
            return False

    def get_line(self) -> Optional[VirtualLine]:
        """라인 조회"""
        return self.line

    def has_line(self) -> bool:
        """라인 존재 여부"""
        return self.line is not None

    def is_line_active(self) -> bool:
        """라인 활성 상태 확인"""
        return self.line is not None and self.line.is_active

    def toggle_line_status(self) -> bool:
        """
        라인 활성/비활성 토글

        Returns:
            bool: 성공 여부
        """
        try:
            if self.line is None:
                return False

            self.line.is_active = not self.line.is_active

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            status = "활성" if self.line.is_active else "비활성"
            logger.info(f"라인 상태 변경: {status}")
            return True

        except Exception as e:
            logger.error(f"라인 상태 변경 실패: {e}")
            return False

    def validate_line_config(self, line_data: dict) -> bool:
        """라인 설정 데이터 유효성 검사"""
        try:
            required_fields = ["name", "start_point", "end_point"]

            # 필수 필드 확인
            for field in required_fields:
                if field not in line_data:
                    logger.error(f"필수 필드 누락: {field}")
                    return False

            # 좌표 유효성 확인
            start_point = line_data["start_point"]
            end_point = line_data["end_point"]

            if not (isinstance(start_point, (list, tuple)) and len(start_point) == 2):
                logger.error(f"시작점 형식 오류: {start_point}")
                return False

            if not (isinstance(end_point, (list, tuple)) and len(end_point) == 2):
                logger.error(f"끝점 형식 오류: {end_point}")
                return False

            return True

        except Exception as e:
            logger.error(f"라인 설정 검증 실패: {e}")
            return False

    def save_to_file(self) -> bool:
        """설정을 파일에 저장"""
        try:
            config_data = {
                "version": "1.0",
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "total_lines": 1 if self.line else 0,
                },
            }

            # 라인이 있는 경우에만 추가
            if self.line:
                config_data["line"] = self.line.to_dict()
            else:
                config_data["line"] = None

            # JSON 파일로 저장
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            logger.info(f"라인 설정 저장됨: {self.config_file_path}")
            return True

        except Exception as e:
            logger.error(f"라인 설정 저장 실패: {e}")
            return False

    def load_from_file(self) -> bool:
        """파일에서 설정 로드"""
        try:
            config_path = Path(self.config_file_path)

            if not config_path.exists():
                logger.info("라인 설정 파일이 없습니다. 빈 상태로 시작합니다.")
                return True

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # 라인 데이터 로드
            line_data = config_data.get("line")
            self.line = None

            if line_data and self.validate_line_config(line_data):
                try:
                    self.line = VirtualLine.from_dict(line_data)
                    logger.info(f"라인 로드됨: {self.line.name}")
                except Exception as e:
                    logger.error(f"라인 로드 실패: {e}")
                    return False
            else:
                logger.info("유효한 라인 설정이 없습니다")

            return True

        except Exception as e:
            logger.error(f"라인 설정 로드 실패: {e}")
            return False

    def backup_config(self) -> bool:
        """설정 백업"""
        try:
            backup_path = f"{self.config_file_path}.backup"

            if Path(self.config_file_path).exists():
                import shutil

                shutil.copy2(self.config_file_path, backup_path)
                logger.info(f"설정 백업됨: {backup_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"설정 백업 실패: {e}")
            return False

    def restore_from_backup(self) -> bool:
        """백업에서 설정 복원"""
        try:
            backup_path = f"{self.config_file_path}.backup"

            if Path(backup_path).exists():
                import shutil

                shutil.copy2(backup_path, self.config_file_path)
                self.load_from_file()
                logger.info("백업에서 설정 복원됨")
                return True

            logger.error("백업 파일을 찾을 수 없습니다")
            return False

        except Exception as e:
            logger.error(f"설정 복원 실패: {e}")
            return False

    def _ensure_config_directory(self) -> None:
        """설정 파일 디렉토리 생성"""
        try:
            config_path = Path(self.config_file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"설정 디렉토리 생성 실패: {e}")

    def get_line_info(self) -> Dict:
        """라인 정보 반환"""
        if self.line is None:
            return {
                "exists": False,
                "is_active": False,
                "name": None,
                "start_point": None,
                "end_point": None,
            }

        return {
            "exists": True,
            "is_active": self.line.is_active,
            "name": self.line.name,
            "start_point": self.line.start_point,
            "end_point": self.line.end_point,
            "color": self.line.color,
            "thickness": self.line.thickness,
        }
