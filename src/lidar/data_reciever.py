import copy
import importlib.util
import os
import socket
import sys
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

from src.lidar.lidar_common_library.datatypes import LidarDatum
from src.lidar.lidar_common_library.imagifier import (
    InferenceLidarImagifier,
    LidarImagifier,
    RGBLidarImagifier,
)


class DataReciever:
    def __init__(self):
        self.imshow_depth_or_hsv = True
        self.depth_color_param = 8000
        self.send_data_to_monitoring = False

        self.data_reciever_folder_path = (
            Path(__file__).parent / "ilidar-api-cpp_mult_V1.0.4"
        )

        self.cfg = self.data_reciever_folder_path / "bin" / "cfg.txt"

        self.multi_thread_read_app = None

        if sys.platform.startswith("win"):
            muiti_thread_read_name = "multi_thread_read_windows"
        else:
            muiti_thread_read_name = "multi_thread_read_ubuntu"

        module_path = os.path.join(
            self.data_reciever_folder_path,
            "bin",
            f"{muiti_thread_read_name}.py",
        )
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(
                f"{muiti_thread_read_name}", module_path
            )
            self.multi_thread_read_app = importlib.util.module_from_spec(spec)
            sys.modules["multi_thread_read_app"] = self.multi_thread_read_app
            spec.loader.exec_module(self.multi_thread_read_app)
        else:
            raise Exception(f"Module not found at path: {module_path}")

        self.ilidar_cfg_read = self.multi_thread_read_app.get_config_file(self.cfg)

        if not self.ilidar_cfg_read:
            exit(-1)
        # print(ilidar_cfg_read)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", self.ilidar_cfg_read.output_dest_port))
        self.sock.settimeout(5)

        while True:

            try:
                data, sender = self.sock.recvfrom(2000)
            except KeyboardInterrupt:
                print("\tCaught Ctrl-C (KeyboardInterrupt)")
                break
            except Exception as e:
                print(
                    "\trecvfrom timeout - maybe multi_thread_read_cmake isn't working"
                )
                sender = None
                continue

            if sender is None:
                pass
            elif len(data) == 130 and sender[1] == self.ilidar_cfg_read.output_src_port:
                self.recv_output_msg = self.multi_thread_read_app.decode_output_msg(
                    data
                )
                print(self.recv_output_msg)
                break

        self.max_col = 320
        self.max_row = 160
        self.raw_size = (
            self.max_col * self.max_row * self.recv_output_msg.ilidar_num * 4
        )
        self.hsv_size = (
            self.max_col * self.max_row * self.recv_output_msg.ilidar_num * 3
        )

        recv_output_msg = self.multi_thread_read_app.decode_output_msg(data)
        cvt_flag = recv_output_msg.cvt_flag

        if cvt_flag != 2:
            self.shm_size = self.ilidar_cfg_read.shm_size * (self.raw_size)
        else:
            self.shm_size = self.ilidar_cfg_read.shm_size * (
                self.raw_size + self.hsv_size
            )

        if sys.platform.startswith("win"):
            self.shm_name = "iLidar_shm_" + "{0:05d}".format(
                self.ilidar_cfg_read.output_src_port
            )
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, size=self.shm_size
            )
        else:
            import sysv_ipc

            self.shm = sysv_ipc.SharedMemory(
                self.ilidar_cfg_read.output_src_port,
                sysv_ipc.IPC_CREAT,
                size=self.shm_size,
            )

        self.nb_of_sensor = self.recv_output_msg.ilidar_num
        self.sensor_sn = self.recv_output_msg.sensor_sn[: self.nb_of_sensor]

    def receive_data(self):

        try:
            data, sender = self.sock.recvfrom(2000)
        except KeyboardInterrupt:
            print("\tCaught Ctrl-C (KeyboardInterrupt)")

        except Exception as e:
            print("\trecvfrom timeout")
            sender = None

        if sender is None:
            pass
        elif len(data) == 130 and sender[1] == self.ilidar_cfg_read.output_src_port:
            recv_output_msg = self.multi_thread_read_app.decode_output_msg(data)
            cvt_flag = recv_output_msg.cvt_flag
            # print(recv_output_msg)

            if cvt_flag == 1:
                shm_offset = recv_output_msg.output_idx * (self.raw_size)

                if sys.platform.startswith("win"):
                    shm_read = self.shm.buf[shm_offset : (shm_offset + self.hsv_size)]
                    img = np.frombuffer(shm_read, dtype=np.uint8)
                else:
                    img = np.frombuffer(
                        self.shm.read(self.hsv_size, shm_offset), dtype=np.uint8
                    )
                # raw_data = img.reshape((recv_output_msg.ilidar_num, self.max_row, self.max_col, 3))
                hsv_data = np.copy(
                    img.reshape(
                        (self.max_row * recv_output_msg.ilidar_num, self.max_col, 3)
                    )
                )
                hsv_raw_data = hsv_data.reshape(
                    (recv_output_msg.ilidar_num, self.max_row, self.max_col, 3)
                )

                imshow_images = {
                    self.sensor_sn[i]: hsv_raw_data[i, :, :, :]
                    for i in range(recv_output_msg.ilidar_num)
                }
                if self.send_data_to_monitoring == True:
                    monitoring_images = copy.deepcopy(imshow_images)
            elif cvt_flag == 0:
                shm_offset = recv_output_msg.output_idx * (self.raw_size)
                if sys.platform.startswith("win"):
                    shm_read = self.shm.buf[shm_offset : (shm_offset + self.raw_size)]
                    img = np.frombuffer(shm_read, dtype=np.uint16)
                else:
                    img = np.frombuffer(
                        self.shm.read(self.raw_size, shm_offset), dtype=np.uint16
                    )
                raw_data = img.reshape(
                    (2, recv_output_msg.ilidar_num, self.max_row, self.max_col)
                )
                depth_data = raw_data[0]
                intensity_data = raw_data[1]
                depth_data_array = [
                    LidarDatum(
                        depth_data[i], intensity_data[i], sensor_info=self.sensor_sn[i]
                    )
                    for i in range(recv_output_msg.ilidar_num)
                ]

                # show_imagifier: LidarImagifier = DepthLidarImagifier() if self.imshow_depth_or_hsv else InferenceLidarImagifier()
                show_imagifier: LidarImagifier = (
                    RGBLidarImagifier()
                    if self.imshow_depth_or_hsv
                    else InferenceLidarImagifier()
                )

                inference_imagifier: LidarImagifier = InferenceLidarImagifier()
                self.imshow_images = {
                    self.sensor_sn[i]: show_imagifier.convert_total(
                        data, self.depth_color_param
                    )
                    for i, data in enumerate(depth_data_array)
                }  # 보여줄 이미지
                if self.send_data_to_monitoring == True:
                    monitoring_images = copy.deepcopy(imshow_images)
                # predict_images = [inference_imagifier.convert_total(data) for data in data_array]    # 모델에 넣을 이미지
            else:  # cvt_flag == 2

                shm_offset = recv_output_msg.output_idx * (
                    self.raw_size + self.hsv_size
                )
                #     )

                shm_read = self.shm.buf[shm_offset : (shm_offset + self.raw_size)]
                depth_img = np.frombuffer(shm_read, dtype=np.uint16)

                shm_read_hsv = self.shm.buf[
                    shm_offset
                    + self.raw_size : (shm_offset + self.raw_size + self.hsv_size)
                ]
                hsv_img = np.frombuffer(shm_read_hsv, dtype=np.uint8)

                # depth_img = np.frombuffer(
                #     self.shm.read(self.raw_size, shm_offset + self.hsv_size),
                #     dtype=np.uint16,
                # )
                # hsv_img = np.frombuffer(
                #     self.shm.read(self.hsv_size, shm_offset), dtype=np.uint8
                # )

                # depth_img2 = depth_img.reshape((2 * recv_output_msg.ilidar_num * self.max_row, self.max_col))
                # hsv_img2 = hsv_img.reshape((recv_output_msg.ilidar_num * self.max_row, self.max_col, 3))
                # cv2.imshow("depth_img2", depth_img2)
                # cv2.imshow("hsv_img2", depth_img2)

                depth_raw_data = depth_img.reshape(
                    (2, recv_output_msg.ilidar_num, self.max_row, self.max_col)
                )
                depth_data = depth_raw_data[0]
                intensity_data = depth_raw_data[1]
                depth_data_array = [
                    LidarDatum(
                        depth_data[i], intensity_data[i], sensor_info=self.sensor_sn[i]
                    )
                    for i in range(recv_output_msg.ilidar_num)
                ]

                hsv_img = hsv_img.reshape(
                    (recv_output_msg.ilidar_num * self.max_row, self.max_col, 3)
                )

                hsv_data = np.copy(
                    hsv_img.reshape(
                        (self.max_row * recv_output_msg.ilidar_num, self.max_col, 3)
                    )
                )
                hsv_raw_data = hsv_data.reshape(
                    (recv_output_msg.ilidar_num, self.max_row, self.max_col, 3)
                )

                # raw_data = hsv_raw_data
                # imshow_images = {self.sensor_sn[i]: raw_data[i, :, :, :] for i in range(recv_output_msg.ilidar_num)}

                raw_data = depth_raw_data

                show_imagifier: LidarImagifier = (
                    RGBLidarImagifier()
                    if self.imshow_depth_or_hsv
                    else InferenceLidarImagifier()
                )

                self.imshow_images = {
                    self.sensor_sn[i]: show_imagifier.convert_total(
                        data, self.depth_color_param
                    )
                    for i, data in enumerate(depth_data_array)
                }  # 보여줄 이미지
                if self.send_data_to_monitoring == True:
                    monitoring_images = copy.deepcopy(imshow_images)

        return self.imshow_images

    def process_data(self):
        pass

    def save_data(self):
        pass

    def initialize(self):
        pass


if __name__ == "__main__":
    data_reciever = DataReciever()

    while True:
        data_reciever.receive_data()

        cv2.imshow("data", data_reciever.imshow_images[data_reciever.sensor_sn[0]])
        cv2.waitKey(1)
