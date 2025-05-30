#!/usr/bin/env python3
"""
Performance test comparing HOGDescriptor vs YOLOX detector.
This script demonstrates the speed and accuracy improvements.
"""

import logging
import time
from typing import List

import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.core.models import DetectionResult
from src.tracking.detector_configs import get_config, list_configs
from src.tracking.engine import ObjectTracker


class HOGDetector:
    """Legacy HOG detector for comparison."""

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect objects using HOG."""
        try:
            boxes, weights = self.hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(32, 32), scale=1.05
            )

            detections = []
            for i, (box, weight) in enumerate(zip(boxes, weights)):
                if weight < self.confidence_threshold:
                    continue

                x, y, w, h = box
                center_x = x + w / 2
                center_y = y + h / 2

                detection = DetectionResult(
                    track_id=-1,
                    bbox=(x, y, w, h),
                    center_point=(center_x, center_y),
                    confidence=float(weight),
                    class_name="person",
                    timestamp=0.0,
                )
                detections.append(detection)

            return detections
        except Exception as e:
            logger.error(f"HOG detection failed: {e}")
            return []


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with some simple shapes."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some colored rectangles to simulate objects
    cv2.rectangle(frame, (100, 100), (200, 300), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(frame, (300, 150), (400, 350), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(frame, (450, 80), (550, 280), (0, 0, 255), -1)  # Red rectangle

    # Add some noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    return frame


def benchmark_detector(detector, frame: np.ndarray, num_iterations: int = 10) -> tuple:
    """Benchmark a detector."""
    times = []
    total_detections = 0

    # Warmup
    for _ in range(3):
        detector.detect_objects(frame)

    # Actual benchmark
    for i in range(num_iterations):
        start_time = time.time()
        detections = detector.detect_objects(frame)
        end_time = time.time()

        times.append(end_time - start_time)
        total_detections += len(detections)

        if (i + 1) % 5 == 0:
            logger.info(f"  Iteration {i + 1}/{num_iterations} completed")

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_detections = total_detections / num_iterations

    return avg_time, std_time, avg_detections


def main():
    """Run performance comparison."""
    logger.info("=== YOLOX vs HOGDescriptor Performance Test ===")

    # Create test frame
    test_frame = create_test_frame()
    logger.info(f"Test frame size: {test_frame.shape}")

    # List available detector configurations
    logger.info("\nAvailable YOLOX configurations:")
    configs = list_configs()
    for name, description in configs.items():
        logger.info(f"  {name}: {description}")

    # Test HOG detector
    logger.info("\n1. Testing HOGDescriptor (legacy)...")
    hog_detector = HOGDetector()
    hog_time, hog_std, hog_detections = benchmark_detector(hog_detector, test_frame)

    # Test YOLOX detectors with different configurations
    yolox_results = {}

    for config_name in ["fast", "balanced", "accurate"]:
        logger.info(f"\n2. Testing YOLOX with '{config_name}' configuration...")

        try:
            # Create tracker with specific config
            tracker = ObjectTracker(detector_config=config_name)
            yolox_time, yolox_std, yolox_detections = benchmark_detector(
                tracker, test_frame
            )
            yolox_results[config_name] = (yolox_time, yolox_std, yolox_detections)

        except Exception as e:
            logger.error(f"Failed to test {config_name} config: {e}")
            yolox_results[config_name] = (float("inf"), 0, 0)

    # Print results
    logger.info("\n=== PERFORMANCE RESULTS ===")
    logger.info(f"HOGDescriptor:")
    logger.info(f"  Average time: {hog_time:.3f}s ± {hog_std:.3f}s")
    logger.info(f"  Average detections: {hog_detections:.1f}")
    logger.info(f"  FPS: {1/hog_time:.1f}")

    for config_name, (yolox_time, yolox_std, yolox_detections) in yolox_results.items():
        if yolox_time != float("inf"):
            speedup = hog_time / yolox_time
            logger.info(f"\nYOLOX ({config_name}):")
            logger.info(f"  Average time: {yolox_time:.3f}s ± {yolox_std:.3f}s")
            logger.info(f"  Average detections: {yolox_detections:.1f}")
            logger.info(f"  FPS: {1/yolox_time:.1f}")
            logger.info(
                f"  Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}"
            )

    # Find best configuration
    best_config = min(yolox_results.items(), key=lambda x: x[1][0])
    if best_config[1][0] != float("inf"):
        logger.info(f"\n=== RECOMMENDATION ===")
        logger.info(f"Best YOLOX configuration: '{best_config[0]}'")
        logger.info(
            f"Performance improvement: {hog_time / best_config[1][0]:.1f}x faster than HOG"
        )


if __name__ == "__main__":
    main()
