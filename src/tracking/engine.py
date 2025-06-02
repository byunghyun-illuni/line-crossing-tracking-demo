"""
Object tracking engine using official OC-SORT implementation with YOLOX detector.
"""

import logging
from typing import List, Optional, Union

import numpy as np

from src.core.models import DetectionResult, TrackingFrame
from src.tracking.detector_configs import DetectorConfig, get_config
from src.tracking.ocsort_tracker.ocsort import OCSort
from src.tracking.yolox_detector import YOLOXDetector

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

            # OCSort 좌표를 원본 이미지 크기로 스케일링 복원
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))

            if len(tracks) > 0:
                tracks[:, :4] *= scale  # x1,y1,x2,y2를 원본 크기로 복원

            # 모든 detection을 DetectionResult로 변환
            tracking_results = []
            for det in detections:
                x, y, w, h = det.bbox
                center_x = x + w / 2
                center_y = y + h / 2

                result = DetectionResult(
                    track_id=-1,  # 기본값
                    bbox=(x, y, w, h),
                    center_point=(center_x, center_y),
                    confidence=det.confidence,
                    class_name=det.class_name,
                    timestamp=0.0,
                )
                tracking_results.append(result)

            # OCSort tracks와 detection 매칭하여 track_id 할당
            for track in tracks:
                x1, y1, x2, y2, track_id = track

                # 가장 높은 IoU를 가진 detection 찾기
                best_match_idx = -1
                best_iou = 0.0

                for i, result in enumerate(tracking_results):
                    if result.track_id > 0:  # 이미 매칭된 detection은 건너뛰기
                        continue

                    det_x, det_y, det_w, det_h = result.bbox
                    det_x1, det_y1 = det_x, det_y
                    det_x2, det_y2 = det_x + det_w, det_y + det_h

                    # IoU 계산
                    inter_x1 = max(x1, det_x1)
                    inter_y1 = max(y1, det_y1)
                    inter_x2 = min(x2, det_x2)
                    inter_y2 = min(y2, det_y2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        track_area = (x2 - x1) * (y2 - y1)
                        det_area = det_w * det_h
                        union_area = track_area + det_area - inter_area

                        if union_area > 0:
                            iou = inter_area / union_area
                            if iou > best_iou and iou > 0.3:  # IoU 임계값 0.3
                                best_iou = iou
                                best_match_idx = i

                # 매칭된 detection에 track ID 할당
                if best_match_idx >= 0:
                    tracking_results[best_match_idx].track_id = int(track_id)

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
