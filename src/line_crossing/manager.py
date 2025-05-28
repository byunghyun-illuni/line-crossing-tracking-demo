"""
가상 라인 관리

가상 라인의 생성, 수정, 삭제, 저장/로드를 담당합니다.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import LINE_CONFIG_FILE
from ..core.enums import CameraViewType
from ..core.models import VirtualLine

logger = logging.getLogger(__name__)


class LineManager:
    """가상 라인 관리 클래스"""

    def __init__(self, config_file_path: str = LINE_CONFIG_FILE):
        self.config_file_path = config_file_path
        self.lines: Dict[str, VirtualLine] = {}
        self.auto_save = True

        # 설정 파일 디렉토리 생성
        self._ensure_config_directory()

        # 기존 설정 로드
        self.load_from_file()

    def create_line(
        self, name: str, start_point: tuple, end_point: tuple, **kwargs
    ) -> str:
        """
        새 라인 생성

        Args:
            name: 라인 이름
            start_point: 시작점 (x, y)
            end_point: 끝점 (x, y)
            **kwargs: 추가 속성 (view_type, thickness, color 등)

        Returns:
            str: 생성된 라인 ID
        """
        try:
            # 라인 ID 생성
            line_id = self._generate_line_id()

            # VirtualLine 객체 생성
            line = VirtualLine(
                line_id=line_id,
                name=name,
                start_point=start_point,
                end_point=end_point,
                is_active=kwargs.get("is_active", True),
                view_type=kwargs.get("view_type", CameraViewType.ENTRANCE),
                thickness=kwargs.get("thickness", 3.0),
                color=kwargs.get("color", (0, 255, 0)),
                direction_config=kwargs.get("direction_config", {}),
            )

            # 라인 유효성 검사
            if not line.validate_points():
                logger.error(
                    f"라인 좌표가 유효하지 않습니다: {start_point} -> {end_point}"
                )
                raise ValueError("Invalid line coordinates")

            # 라인 저장
            self.lines[line_id] = line

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            logger.info(f"라인 생성됨: {name} ({line_id})")
            return line_id

        except Exception as e:
            logger.error(f"라인 생성 실패: {e}")
            raise

    def update_line(self, line_id: str, **kwargs) -> bool:
        """
        라인 업데이트

        Args:
            line_id: 라인 ID
            **kwargs: 업데이트할 속성들

        Returns:
            bool: 성공 여부
        """
        try:
            if line_id not in self.lines:
                logger.error(f"라인을 찾을 수 없습니다: {line_id}")
                return False

            line = self.lines[line_id]

            # 속성 업데이트
            for key, value in kwargs.items():
                if hasattr(line, key):
                    setattr(line, key, value)
                else:
                    logger.warning(f"알 수 없는 속성: {key}")

            # 좌표가 변경된 경우 유효성 검사
            if "start_point" in kwargs or "end_point" in kwargs:
                if not line.validate_points():
                    logger.error(f"업데이트된 라인 좌표가 유효하지 않습니다: {line_id}")
                    return False

                # 방향 설정 재계산
                line.direction_config = line._auto_detect_direction_config()

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            logger.info(f"라인 업데이트됨: {line_id}")
            return True

        except Exception as e:
            logger.error(f"라인 업데이트 실패: {e}")
            return False

    def delete_line(self, line_id: str) -> bool:
        """
        라인 삭제

        Args:
            line_id: 라인 ID

        Returns:
            bool: 성공 여부
        """
        try:
            if line_id not in self.lines:
                logger.error(f"라인을 찾을 수 없습니다: {line_id}")
                return False

            line_name = self.lines[line_id].name
            del self.lines[line_id]

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            logger.info(f"라인 삭제됨: {line_name} ({line_id})")
            return True

        except Exception as e:
            logger.error(f"라인 삭제 실패: {e}")
            return False

    def get_line(self, line_id: str) -> Optional[VirtualLine]:
        """라인 조회"""
        return self.lines.get(line_id)

    def get_all_lines(self) -> Dict[str, VirtualLine]:
        """모든 라인 조회"""
        return self.lines.copy()

    def get_active_lines(self) -> Dict[str, VirtualLine]:
        """활성 라인만 조회"""
        return {line_id: line for line_id, line in self.lines.items() if line.is_active}

    def toggle_line_status(self, line_id: str) -> bool:
        """
        라인 활성/비활성 토글

        Args:
            line_id: 라인 ID

        Returns:
            bool: 성공 여부
        """
        try:
            if line_id not in self.lines:
                return False

            self.lines[line_id].is_active = not self.lines[line_id].is_active

            # 자동 저장
            if self.auto_save:
                self.save_to_file()

            status = "활성" if self.lines[line_id].is_active else "비활성"
            logger.info(f"라인 상태 변경: {line_id} -> {status}")
            return True

        except Exception as e:
            logger.error(f"라인 상태 변경 실패: {e}")
            return False

    def validate_line_config(self, line_data: dict) -> bool:
        """라인 설정 데이터 유효성 검사"""
        try:
            required_fields = ["line_id", "name", "start_point", "end_point"]

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
            # 라인 데이터를 딕셔너리로 변환
            lines_data = {
                line_id: line.to_dict() for line_id, line in self.lines.items()
            }

            config_data = {
                "version": "1.0",
                "lines": lines_data,
                "metadata": {
                    "total_lines": len(self.lines),
                    "active_lines": len(self.get_active_lines()),
                },
            }

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
            lines_data = config_data.get("lines", {})
            self.lines = {}

            for line_id, line_data in lines_data.items():
                if self.validate_line_config(line_data):
                    try:
                        line = VirtualLine.from_dict(line_data)
                        self.lines[line_id] = line
                    except Exception as e:
                        logger.error(f"라인 로드 실패: {line_id} - {e}")
                        continue
                else:
                    logger.warning(f"유효하지 않은 라인 설정 무시: {line_id}")

            logger.info(f"라인 설정 로드됨: {len(self.lines)}개 라인")
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

    def _generate_line_id(self) -> str:
        """고유한 라인 ID 생성"""
        return f"line_{uuid.uuid4().hex[:8]}"

    def _validate_line_data(self, data: dict) -> bool:
        """라인 데이터 유효성 검사 (내부용)"""
        return self.validate_line_config(data)

    def _ensure_config_directory(self) -> None:
        """설정 파일 디렉토리 생성"""
        try:
            config_path = Path(self.config_file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"설정 디렉토리 생성 실패: {e}")

    def get_line_count(self) -> int:
        """총 라인 수 반환"""
        return len(self.lines)

    def get_active_line_count(self) -> int:
        """활성 라인 수 반환"""
        return len(self.get_active_lines())

    def clear_all_lines(self) -> bool:
        """모든 라인 삭제"""
        try:
            self.lines.clear()

            if self.auto_save:
                self.save_to_file()

            logger.info("모든 라인이 삭제되었습니다")
            return True

        except Exception as e:
            logger.error(f"라인 전체 삭제 실패: {e}")
            return False
