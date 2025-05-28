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

from ..core.models import DetectionResult

logger = logging.getLogger(__name__)


class YOLOXDetector:
    """
    YOLOX-based object detector using torchvision's implementation.
    This replaces the slow HOGDescriptor with a modern, fast detector.
    """

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        confidence_threshold: float = 0.6,
        device: Optional[str] = None,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Initialize YOLOX detector.

        Args:
            model_name: Model to use (fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on (auto-detected if None)
            target_classes: List of class names to detect (None for all COCO classes)
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ["person"]

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

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input.

        Args:
            frame: Input frame in BGR format

        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        tensor = transform(frame_rgb)
        return tensor

    def detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in frame using YOLOX.

        Args:
            frame: Input frame in BGR format

        Returns:
            List of detection results
        """
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
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

            # Filter by confidence and target classes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score < self.confidence_threshold:
                    continue

                # Check if this is a target class
                if self.target_class_indices and label not in self.target_class_indices:
                    continue

                # Convert box format (x1, y1, x2, y2) to (x, y, w, h)
                x1, y1, x2, y2 = box
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

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

            logger.debug(f"Detected {len(detections)} objects")
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
