"""
Detector configuration presets based on MMTracking's approach.
Provides easy-to-use configurations for different detection models.
"""

from typing import Any, Dict, List, Optional


class DetectorConfig:
    """Base detector configuration class."""

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.6,
        target_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ["person"]
        self.device = device
        self.extra_params = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "target_classes": self.target_classes,
            "device": self.device,
            **self.extra_params,
        }


# Predefined configurations based on MMTracking's OC-SORT configs

# Fast configuration - optimized for speed
FAST_CONFIG = DetectorConfig(
    model_name="fasterrcnn_mobilenet_v3_large_fpn",
    confidence_threshold=0.5,
    target_classes=["person"],
    description="Fast detector optimized for real-time performance",
)

# Balanced configuration - good balance of speed and accuracy
BALANCED_CONFIG = DetectorConfig(
    model_name="fasterrcnn_resnet50_fpn",
    confidence_threshold=0.6,
    target_classes=["person"],
    description="Balanced detector for good speed and accuracy",
)

# Accurate configuration - optimized for accuracy
ACCURATE_CONFIG = DetectorConfig(
    model_name="retinanet_resnet50_fpn",
    confidence_threshold=0.5,
    target_classes=["person"],
    description="Accurate detector optimized for precision",
)

# 새로운 고성능 설정 추가
HIGH_PRECISION_CONFIG = DetectorConfig(
    model_name="fasterrcnn_resnet50_fpn",
    confidence_threshold=0.4,
    target_classes=["person"],
    description="High precision detector for crowded scenes",
)

# 복잡한 환경용 설정
CROWDED_SCENE_CONFIG = DetectorConfig(
    model_name="retinanet_resnet50_fpn",
    confidence_threshold=0.25,
    target_classes=["person"],
    description="Optimized for crowded and complex scenes like malls (임계값 0.25)",
)

# Multi-class configuration - detects multiple object types
MULTICLASS_CONFIG = DetectorConfig(
    model_name="fasterrcnn_resnet50_fpn",
    confidence_threshold=0.6,
    target_classes=["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    description="Multi-class detector for various objects",
)

# COCO configuration - detects all COCO classes
COCO_CONFIG = DetectorConfig(
    model_name="fasterrcnn_resnet50_fpn",
    confidence_threshold=0.6,
    target_classes=None,  # All COCO classes
    description="Full COCO detector for all object classes",
)


def get_config(config_name: str) -> DetectorConfig:
    """
    Get a predefined detector configuration.

    Args:
        config_name: Name of the configuration

    Returns:
        DetectorConfig instance

    Raises:
        ValueError: If config_name is not found
    """
    configs = {
        "fast": FAST_CONFIG,
        "balanced": BALANCED_CONFIG,
        "accurate": ACCURATE_CONFIG,
        "high_precision": HIGH_PRECISION_CONFIG,
        "crowded_scene": CROWDED_SCENE_CONFIG,
        "multiclass": MULTICLASS_CONFIG,
        "coco": COCO_CONFIG,
    }

    if config_name.lower() not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown config '{config_name}'. Available configs: {available}"
        )

    return configs[config_name.lower()]


def list_configs() -> Dict[str, str]:
    """
    List all available detector configurations.

    Returns:
        Dictionary mapping config names to descriptions
    """
    return {
        "fast": "Fast detector optimized for real-time performance",
        "balanced": "Balanced detector for good speed and accuracy",
        "accurate": "Accurate detector optimized for precision (낮은 임계값)",
        "high_precision": "High precision detector for crowded scenes (임계값 0.4)",
        "crowded_scene": "Optimized for crowded and complex scenes like malls (임계값 0.25)",
        "multiclass": "Multi-class detector for various objects",
        "coco": "Full COCO detector for all object classes",
    }
