"""
Object tracking engine using official OC-SORT implementation with YOLOX detector.
"""

import logging
from typing import List, Optional, Union

import cv2
import numpy as np

from ..core.models import DetectionResult, TrackingFrame
from .detector_configs import DetectorConfig, get_config
from .ocsort_tracker.ocsort import OCSort
from .yolox_detector import YOLOXDetector

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Object tracker using official OC-SORT implementation with YOLOX detector.
    This replaces the slow HOGDescriptor with a modern, fast detection system.
    """

    def __init__(
        self,
        det_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        detector_config: Union[str, DetectorConfig, None] = None,
        detector_model: Optional[str] = None,
        detector_confidence: Optional[float] = None,
        target_classes: Optional[List[str]] = None,
        device: Optional[str] = None,
        enable_image_enhancement: bool = False,
        nms_iou_threshold: Optional[float] = None,
    ):
        """
        Initialize OC-SORT tracker with YOLOX detector.

        Args:
            det_thresh: Detection confidence threshold for tracker
            max_age: Maximum number of frames to keep alive a track without associated detections
            min_hits: Minimum number of associated detections before track is initialised
            iou_threshold: Minimum IOU for match
            delta_t: Time step for velocity calculation
            asso_func: Association function ("iou", "giou", "ciou", "diou", "ct_dist")
            inertia: Inertia factor for velocity
            use_byte: Whether to use ByteTrack association
            detector_config: Predefined detector config name or DetectorConfig instance
            detector_model: Model name for YOLOX detector (overrides config)
            detector_confidence: Confidence threshold for detector (overrides config)
            target_classes: List of class names to detect (overrides config)
            device: Device for detector inference (overrides config)
            enable_image_enhancement: Whether to enable image enhancement
            nms_iou_threshold: Non-maximum suppression IOU threshold
        """
        self.tracker = OCSort(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte,
        )

        # Handle detector configuration
        if detector_config is not None:
            if isinstance(detector_config, str):
                config = get_config(detector_config)
            else:
                config = detector_config

            # Use config values, but allow overrides
            detector_params = {
                "model_name": detector_model or config.model_name,
                "confidence_threshold": detector_confidence
                or config.confidence_threshold,
                "target_classes": target_classes or config.target_classes,
                "device": device or config.device,
                "enable_image_enhancement": enable_image_enhancement,
                "nms_iou_threshold": nms_iou_threshold,
            }
        else:
            # Use default values or provided parameters
            detector_params = {
                "model_name": detector_model or "fasterrcnn_resnet50_fpn",
                "confidence_threshold": detector_confidence or 0.6,
                "target_classes": target_classes or ["person"],
                "device": device,
                "enable_image_enhancement": enable_image_enhancement,
                "nms_iou_threshold": nms_iou_threshold,
            }

        # Initialize YOLOX detector (replaces HOGDescriptor)
        self.detector = YOLOXDetector(**detector_params)

        # Warm up the detector for faster first inference
        self.detector.warmup()

        logger.info("ObjectTracker initialized with OC-SORT and YOLOX detector")
        logger.info(f"Detector config: {detector_params}")

    def detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in frame using YOLOX detector.

        Args:
            frame: Input frame

        Returns:
            List of detection results
        """
        return self.detector.detect_objects(frame)

    def track_frame(
        self, frame: np.ndarray, detections: Optional[List[DetectionResult]] = None
    ) -> TrackingFrame:
        """
        Track objects in a single frame.

        Args:
            frame: Input frame
            detections: Pre-computed detections (if None, will detect automatically)

        Returns:
            TrackingFrame with tracking results
        """
        try:
            # Get detections if not provided
            if detections is None:
                detections = self.detect_objects(frame)

            # Convert detections to OC-SORT format (x1, y1, x2, y2, confidence)
            if len(detections) > 0:
                dets = []
                for det in detections:
                    x, y, w, h = det.bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    dets.append([x1, y1, x2, y2, det.confidence])
                dets = np.array(dets)
            else:
                dets = np.empty((0, 5))

            # Get frame info
            img_info = (frame.shape[0], frame.shape[1])  # height, width
            img_size = (frame.shape[1], frame.shape[0])  # width, height

            # Update tracker
            tracks = self.tracker.update(dets, img_info, img_size)

            # Convert tracks back to our format
            tracking_results = []
            for track in tracks:
                x1, y1, x2, y2, track_id = track

                # Calculate bbox in our format (x, y, w, h)
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                center_x = x + w / 2
                center_y = y + h / 2

                # Find corresponding detection for class info
                class_name = "person"
                confidence = 0.5

                # Try to match with original detections
                for det in detections:
                    det_x, det_y, det_w, det_h = det.bbox
                    det_x1, det_y1, det_x2, det_y2 = (
                        det_x,
                        det_y,
                        det_x + det_w,
                        det_y + det_h,
                    )

                    # Simple overlap check
                    if (
                        abs(det_x1 - x1) < 50
                        and abs(det_y1 - y1) < 50
                        and abs(det_x2 - x2) < 50
                        and abs(det_y2 - y2) < 50
                    ):
                        class_name = det.class_name
                        confidence = det.confidence
                        break

                result = DetectionResult(
                    track_id=int(track_id),
                    bbox=(x, y, w, h),
                    center_point=(center_x, center_y),
                    confidence=confidence,
                    class_name=class_name,
                    timestamp=0.0,  # Will be set by caller
                )
                tracking_results.append(result)

            return TrackingFrame(
                frame_id=getattr(self.tracker, "frame_count", 0),
                timestamp=0.0,
                detections=tracking_results,
            )

        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            return TrackingFrame(
                frame_id=getattr(self.tracker, "frame_count", 0),
                timestamp=0.0,
                detections=[],
            )

    def reset(self):
        """Reset the tracker."""
        self.tracker = OCSort(
            det_thresh=self.tracker.det_thresh,
            max_age=self.tracker.max_age,
            min_hits=self.tracker.min_hits,
            iou_threshold=self.tracker.iou_threshold,
            delta_t=self.tracker.delta_t,
            asso_func="iou",  # Reset to default
            inertia=self.tracker.inertia,
            use_byte=self.tracker.use_byte,
        )
        logger.info("Tracker reset")
