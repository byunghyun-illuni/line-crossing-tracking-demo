"""
YOLOX-based object detector for improved performance over HOGDescriptor.
Based on MMTracking's OC-SORT implementation approach.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import detection
from torchvision.ops import nms  # NMS 추가

from src.core.models import DetectionResult

logger = logging.getLogger(__name__)


class YOLOXDetector:
    """
    YOLOX-based object detector using torchvision's implementation.
    """

    # 검출 관련 상수들
    DEFAULT_MIN_CONFIDENCE = 0.2
    DEFAULT_CONFIDENCE_MARGIN = 0.2
    DEFAULT_MIN_BOX_WIDTH = 10
    DEFAULT_MIN_BOX_HEIGHT = 20
    DEFAULT_NMS_IOU_THRESHOLD = 0.4

    # 이미지 처리 관련 상수들
    SMALL_IMAGE_THRESHOLD = 800
    LARGE_IMAGE_THRESHOLD = 1920
    SCALE_FACTOR = 2.0
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    SHARPEN_WEIGHT = 0.3
    ENHANCE_WEIGHT = 0.7

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        confidence_threshold: float = 0.6,
        device: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
        # 핵심 설정만 파라미터로 노출
        enable_image_enhancement: bool = False,
        nms_iou_threshold: Optional[float] = None,
    ):
        """
        Initialize YOLOX detector.

        Args:
            model_name: Model to use (fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on (auto-detected if None)
            target_classes: List of class names to detect (None for all COCO classes)
            enable_image_enhancement: Whether to apply image enhancement for better small object detection
            nms_iou_threshold: IoU threshold for NMS (uses default if None)
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ["person"]
        self.enable_image_enhancement = enable_image_enhancement
        self.nms_iou_threshold = nms_iou_threshold or self.DEFAULT_NMS_IOU_THRESHOLD

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        # COCO class names (index 0 is background)
        self.coco_classes = [
            "__background__",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "N/A",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "N/A",
            "backpack",
            "umbrella",
            "N/A",
            "N/A",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "N/A",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "N/A",
            "dining table",
            "N/A",
            "N/A",
            "toilet",
            "N/A",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "N/A",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        # Create mapping for target classes
        self.target_class_indices = []
        for class_name in self.target_classes:
            if class_name in self.coco_classes:
                self.target_class_indices.append(self.coco_classes.index(class_name))

        logger.info(f"YOLOXDetector initialized with model: {model_name}")
        logger.info(f"Target classes: {self.target_classes}")
        logger.info(f"Target class indices: {self.target_class_indices}")

    def _load_model(self, model_name: str):
        """Load the detection model."""
        try:
            if model_name == "fasterrcnn_resnet50_fpn":
                model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
                model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
            elif model_name == "retinanet_resnet50_fpn":
                model = detection.retinanet_resnet50_fpn(pretrained=True)
            else:
                logger.warning(
                    f"Unknown model {model_name}, falling back to fasterrcnn_resnet50_fpn"
                )
                model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[torch.Tensor, float]:
        """
        Preprocess frame for model input with enhanced scaling.

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple of (preprocessed tensor, scale_factor)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 원본 크기 저장
        original_height, original_width = frame_rgb.shape[:2]
        scale_factor = 1.0

        # 해상도가 낮으면 업스케일링 (작은 객체 감지 개선)
        if (
            original_width < self.SMALL_IMAGE_THRESHOLD
            or original_height < self.SMALL_IMAGE_THRESHOLD
        ):
            # 작은 이미지는 확대
            scale_factor = self.SCALE_FACTOR
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            frame_rgb = cv2.resize(
                frame_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
        elif (
            original_width > self.LARGE_IMAGE_THRESHOLD
            or original_height > self.LARGE_IMAGE_THRESHOLD
        ):
            # 너무 큰 이미지는 축소 (처리 속도 개선)
            scale_factor = min(
                self.LARGE_IMAGE_THRESHOLD / original_width,
                self.LARGE_IMAGE_THRESHOLD / original_height,
            )
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            frame_rgb = cv2.resize(
                frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        # Convert to tensor and normalize
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        tensor = transform(frame_rgb)
        return tensor, scale_factor

    def detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in frame using YOLOX with enhanced processing.

        Args:
            frame: Input frame in BGR format

        Returns:
            List of detection results
        """
        try:
            # 이미지 품질 향상 (선택적 적용)
            if self.enable_image_enhancement:
                enhanced_frame = self._enhance_small_objects(frame)
            else:
                enhanced_frame = frame

            # Preprocess frame (이제 scale_factor도 반환)
            input_tensor, scale_factor = self._preprocess_frame(enhanced_frame)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model(input_tensor)

            # Process predictions
            detections = []
            pred = predictions[0]

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            # 커스텀 NMS 적용
            boxes, scores, labels = self._apply_custom_nms(boxes, scores, labels)

            # 더 관대한 필터링을 위한 개선된 로직
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # 동적 신뢰도 임계값 계산
                min_threshold = max(
                    self.DEFAULT_MIN_CONFIDENCE,
                    self.confidence_threshold - self.DEFAULT_CONFIDENCE_MARGIN,
                )
                if score < min_threshold:
                    continue

                # Check if this is a target class
                if self.target_class_indices and label not in self.target_class_indices:
                    continue

                # Convert box format (x1, y1, x2, y2) to (x, y, w, h)
                # 중요: scale_factor로 원본 이미지 좌표로 변환
                x1, y1, x2, y2 = box / scale_factor
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                # 최소 바운딩 박스 크기 검증
                if w < self.DEFAULT_MIN_BOX_WIDTH or h < self.DEFAULT_MIN_BOX_HEIGHT:
                    continue

                # 이미지 경계 체크 (원본 이미지 크기 기준)
                orig_height, orig_width = frame.shape[:2]
                if x < 0 or y < 0 or x + w > orig_width or y + h > orig_height:
                    continue

                # Calculate center point
                center_x = x + w / 2
                center_y = y + h / 2

                # Get class name
                class_name = (
                    self.coco_classes[label]
                    if label < len(self.coco_classes)
                    else "unknown"
                )

                detection = DetectionResult(
                    track_id=-1,  # Will be assigned by tracker
                    bbox=(x, y, w, h),
                    center_point=(center_x, center_y),
                    confidence=float(score),
                    class_name=class_name,
                    timestamp=0.0,  # Will be set by caller
                )
                detections.append(detection)

            logger.debug(
                f"Detected {len(detections)} objects (scale_factor: {scale_factor:.2f})"
            )
            return detections

        except Exception as e:
            logger.error(f"YOLOX detection failed: {e}")
            return []

    def warmup(self, input_size: Tuple[int, int] = (640, 480)):
        """
        Warm up the model with a dummy input for faster first inference.

        Args:
            input_size: Input size (width, height) for warmup
        """
        try:
            logger.info("Warming up YOLOX detector...")
            dummy_frame = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)

            # Run a few warmup iterations
            for _ in range(3):
                self.detect_objects(dummy_frame)

            logger.info("YOLOX detector warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _apply_custom_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        iou_threshold: Optional[float] = None,
    ) -> tuple:
        """
        커스텀 NMS 적용으로 겹치는 감지 결과 정리

        Args:
            boxes: 바운딩 박스 배열 (N, 4)
            scores: 신뢰도 점수 (N,)
            labels: 클래스 라벨 (N,)
            iou_threshold: IoU 임계값 (None이면 인스턴스 설정 사용)

        Returns:
            필터링된 (boxes, scores, labels)
        """
        if len(boxes) == 0:
            return boxes, scores, labels

        # Use instance setting or provided threshold
        threshold = iou_threshold or self.nms_iou_threshold

        # Convert to torch tensors
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()

        # Apply NMS
        keep_indices = nms(boxes_tensor, scores_tensor, threshold)
        keep_indices = keep_indices.numpy()

        return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

    def _enhance_small_objects(self, frame: np.ndarray) -> np.ndarray:
        """
        작은 객체 감지 향상을 위한 이미지 전처리

        Args:
            frame: 입력 프레임

        Returns:
            향상된 프레임
        """
        # 대비 향상
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_TILE_SIZE
        )
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 샤프닝 필터 적용
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 원본과 블렌딩
        result = cv2.addWeighted(
            enhanced, self.ENHANCE_WEIGHT, sharpened, self.SHARPEN_WEIGHT, 0
        )

        return result
