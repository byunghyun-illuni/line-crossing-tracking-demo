#!/usr/bin/env python3
"""
ONNX 모델 기반 라이다 객체 검출 테스트
실시간 프레임별 추론 확인용
"""

import copy
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from Crypto.Cipher import AES

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lidar.data_reciever import DataReciever


class LidarONNXDetector:
    """라이다 ONNX 모델 기반 객체 검출기"""

    def __init__(self, model_type=1):
        """
        Args:
            model_type: 1=암호화된 ONNX(.o.shas), 2=일반 ONNX(.onnx)
        """
        self.model_type = model_type
        self.session = None
        self.lidar_data_receiver = DataReciever()
        self.iou_threshold = 0.5
        self.detect_confidence = 0.3

        # 모델 초기화
        self.init_model()

    def decrypt_model(self, enc_file_path, key):
        """암호화된 모델 복호화"""
        with open(enc_file_path, "rb") as f:
            nonce = f.read(16)
            tag = f.read(16)
            ciphertext = f.read()

        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        decrypted = cipher.decrypt_and_verify(ciphertext, tag)
        return decrypted

    def init_model(self):
        """ONNX 모델 초기화"""
        if self.model_type == 1:  # onnx encryption O
            onnx_files = [f for f in os.listdir(".") if f.endswith(".o.shas")]
            if onnx_files:
                latest_onnx_file = max(onnx_files, key=os.path.getmtime)
                print(f"ONNX Model (암호화): {latest_onnx_file}")
            else:
                print("암호화된 ONNX 파일(.o.shas)이 없습니다.")
                return False

            key = b"M0d3lS3cur3K3y!!"
            decrypted_model = self.decrypt_model(latest_onnx_file, key)

            self.session = ort.InferenceSession(
                decrypted_model,
                sess_options=ort.SessionOptions(),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        elif self.model_type == 2:  # onnx encryption X
            onnx_files = [f for f in os.listdir(".") if f.endswith(".onnx")]
            if onnx_files:
                latest_onnx_file = max(onnx_files, key=os.path.getmtime)
                print(f"ONNX Model (일반): {latest_onnx_file}")
            else:
                print("일반 ONNX 파일(.onnx)이 없습니다.")
                return False

            self.session = ort.InferenceSession(
                latest_onnx_file,
                sess_options=ort.SessionOptions(),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            print("지원하지 않는 모델 타입입니다.")
            return False

        print("사용 가능한 Providers:", ort.get_available_providers())

        # 모델 입력/출력 정보 확인
        if self.session:
            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]
            print(
                f"모델 입력: {input_info.name}, 형태: {input_info.shape}, 타입: {input_info.type}"
            )
            print(
                f"모델 출력: {output_info.name}, 형태: {output_info.shape}, 타입: {output_info.type}"
            )

        print("모델 초기화 완료")
        return True

    def preprocess_frame(self, lidar_images):
        """라이다 데이터 전처리 (기존 프로젝트 방식 적용)"""
        if not lidar_images:
            return None

        # 첫 번째 센서 데이터 사용
        sensor_id = list(lidar_images.keys())[0]
        frame = lidar_images[sensor_id]

        if frame is None:
            return None

        print(f"입력 프레임 형태: {frame.shape}, 타입: {frame.dtype}")

        # 기존 프로젝트에서는 raw_data가 (2, ilidar_num, max_row, max_col) 형태
        # depth_data[0], intensity_data[1] 구조

        if len(frame.shape) == 2:
            # 2D 깊이 맵 (H, W) -> depth/intensity 구조로 변환
            depth_data = frame
            intensity_data = frame.copy()  # 임시로 같은 데이터 사용

            # (2, 1, H, W) 형태로 만들기
            raw_data = np.stack([depth_data, intensity_data], axis=0)
            raw_data = np.expand_dims(raw_data, axis=1)

        elif len(frame.shape) == 3:
            # 3D 데이터인 경우
            if frame.shape[2] == 2:
                # depth + intensity 채널 (H, W, 2)
                depth_data = frame[:, :, 0]
                intensity_data = frame[:, :, 1]
            else:
                # 첫 번째 채널을 depth로 사용
                depth_data = frame[:, :, 0]
                intensity_data = depth_data.copy()

            raw_data = np.stack([depth_data, intensity_data], axis=0)
            raw_data = np.expand_dims(raw_data, axis=1)

        else:
            print(f"지원하지 않는 프레임 형태: {frame.shape}")
            return None

        # 기존 프로젝트 방식: transpose (2, ilidar_num, H, W) -> (ilidar_num, 2, H, W)
        input_data = np.transpose(raw_data, axes=(1, 0, 2, 3)).astype(dtype=np.float32)

        print(f"전처리 후 형태: {input_data.shape} (센서수, 채널, H, W)")
        return input_data

    def nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression (기존 프로젝트 방식)"""
        if len(boxes) == 0:
            return []

        # 좌표 추출
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # 면적 계산
        areas = (x2 - x1) * (y2 - y1)

        # 점수순으로 정렬
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 교집합 계산
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # IoU 계산
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # IoU 임계값보다 작은 것들만 유지
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect_objects(self, lidar_images):
        """객체 검출 실행 (기존 프로젝트 방식)"""
        if self.session is None:
            print("모델이 초기화되지 않았습니다.")
            return []

        # 전처리
        input_data = self.preprocess_frame(lidar_images)
        if input_data is None:
            return []

        try:
            # ONNX 추론 실행 (기존 프로젝트 방식)
            outputs = self.session.run(["output"], {"input": input_data})
            raw_output = outputs[0]  # shape: (batch, 5, num_detections)

            print(f"ONNX 출력 형태: {raw_output.shape}")

            # YOLO 형태 좌표 변환: center_x,center_y,w,h -> x1,y1,x2,y2
            xyxy_output = copy.deepcopy(raw_output)
            xyxy_output[:, 0, :] = raw_output[:, 0, :] - raw_output[:, 2, :] / 2  # x1
            xyxy_output[:, 1, :] = raw_output[:, 1, :] - raw_output[:, 3, :] / 2  # y1
            xyxy_output[:, 2, :] = raw_output[:, 0, :] + raw_output[:, 2, :] / 2  # x2
            xyxy_output[:, 3, :] = raw_output[:, 1, :] + raw_output[:, 3, :] / 2  # y2

            prediction = []
            for batch in range(raw_output.shape[0]):
                current_output = np.transpose(xyxy_output[batch])  # (num_detections, 5)

                # 신뢰도 필터링
                current_output = current_output[
                    current_output[:, 4] > self.detect_confidence
                ]

                if len(current_output) > 0:
                    # NMS 적용
                    boxes = current_output[:, :4]
                    scores = current_output[:, 4]
                    nms_indices = self.nms(boxes, scores, self.iou_threshold)
                    current_output = current_output[nms_indices]

                prediction.append(current_output)

            return prediction

        except Exception as e:
            print(f"추론 중 오류 발생: {e}")
            import traceback

            traceback.print_exc()
            return []

    def draw_detections(self, frame, detections):
        """검출 결과를 프레임에 그리기"""
        annotated_frame = frame.copy()

        if len(detections) == 0 or len(detections[0]) == 0:
            return annotated_frame

        # 첫 번째 배치 결과 사용
        batch_detections = detections[0]

        for detection in batch_detections:
            x1, y1, x2, y2, conf = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 바운딩 박스 그리기
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 라벨 그리기
            label = f"person: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                annotated_frame,
                (x1, y1 - 25),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return annotated_frame

    def run_detection_test(self):
        """라이다 검출 테스트 실행"""
        if self.session is None:
            print("모델 초기화 실패")
            return

        print("라이다 객체 검출 테스트 시작...")
        print("ESC 키를 누르면 종료됩니다")
        print("=" * 50)

        frame_count = 0
        fps_start_time = time.time()

        try:
            while True:
                # 라이다 데이터 수신
                lidar_images = self.lidar_data_receiver.receive_data()

                if lidar_images and self.lidar_data_receiver.sensor_sn:
                    # 첫 번째 센서의 이미지 가져오기
                    sensor_id = self.lidar_data_receiver.sensor_sn[0]
                    frame = lidar_images.get(sensor_id)

                    if frame is not None:
                        frame_count += 1

                        # 객체 검출
                        start_time = time.time()
                        detections = self.detect_objects(lidar_images)
                        inference_time = time.time() - start_time

                        # 검출 결과 출력
                        if frame_count % 30 == 0:  # 1초마다 출력
                            current_time = time.time()
                            fps = frame_count / (current_time - fps_start_time)

                            total_detections = sum(len(d) for d in detections)
                            print(
                                f"Frame {frame_count:4d} | "
                                f"검출: {total_detections:2d}개 | "
                                f"추론시간: {inference_time*1000:5.1f}ms | "
                                f"FPS: {fps:5.1f}"
                            )

                            # 검출된 객체 상세 정보
                            if total_detections > 0:
                                for batch_idx, batch_dets in enumerate(detections):
                                    for i, det in enumerate(batch_dets):
                                        x1, y1, x2, y2, conf = det
                                        print(
                                            f"  배치{batch_idx} 객체{i+1}: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}), conf={conf:.3f}"
                                        )

                        # 검출 결과 시각화
                        annotated_frame = self.draw_detections(frame, detections)

                        # 프레임 표시
                        cv2.imshow("Lidar Object Detection", annotated_frame)

                    else:
                        print("Warning: 라이다 프레임 데이터가 없습니다")
                else:
                    print("Warning: 라이다 데이터를 받지 못했습니다")

                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

        except KeyboardInterrupt:
            print("\n사용자 중단")
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # 정리 작업
            cv2.destroyAllWindows()

            # 최종 통계
            if frame_count > 0:
                total_time = time.time() - fps_start_time
                avg_fps = frame_count / total_time
                print(f"\n최종 통계:")
                print(f"총 프레임: {frame_count}")
                print(f"평균 FPS: {avg_fps:.1f}")
                print(f"총 실행 시간: {total_time:.1f}초")


def main():
    """메인 함수"""
    print("라이다 ONNX 객체 검출 테스트")
    print("=" * 40)

    # 모델 타입 선택
    model_type = 1  # 1: 암호화된 ONNX, 2: 일반 ONNX

    # 검출기 초기화
    detector = LidarONNXDetector(model_type=model_type)

    # 검출 테스트 실행
    detector.run_detection_test()


if __name__ == "__main__":
    main()
