#include <thread>
#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <condition_variable>	// Data synchronization
#include <mutex>				// Data synchronization
#include <queue>				// Data synchronization
#include <ctime>

#include <iostream>
#pragma warning(disable: 4996)

#if defined(_WIN32) || defined(_WIN64)
// Windows headers
#include <conio.h>
#include <tchar.h>
#else
// Non-windows headers
/* Assume that any non-Windows platform uses POSIX-style sockets instead */
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

// Include ilidar library
#include "../src/ilidar.hpp"

// CPP version
#define MULT_THREAD_READ_VERSION_MAJOR ((uint8_t)1)
#define MULT_THREAD_READ_VERSION_MINOR ((uint8_t)0)
#define MULT_THREAD_READ_VERSION_PATCH ((uint8_t)4)

static const uint8_t version[] = {
	MULT_THREAD_READ_VERSION_PATCH,
	MULT_THREAD_READ_VERSION_MINOR,
	MULT_THREAD_READ_VERSION_MAJOR
};

// cv::Mat definition for global data pointer
static cv::Mat lidar_img_depth[iTFS::max_device];
static cv::Mat lidar_img_intensity[iTFS::max_device];

static bool lidar_img_ready[iTFS::max_device];
static std::chrono::time_point<std::chrono::system_clock> lidar_img_time[iTFS::max_device];

static std::mutex lidar_img_mutex;

static bool stream_ready = false;

// cv::Mat definition for output data holder
static cv::Mat lidar_output_raw;
static cv::Mat output_raw;
static cv::Mat output_hsv;

// Synchronization variables
static std::condition_variable	lidar_cv[iTFS::max_device];
static std::mutex				lidar_cv_mutex[iTFS::max_device];
static std::queue<int>			lidar_q[iTFS::max_device];

static std::condition_variable	output_cv;
static std::mutex				output_cv_mutex;
static std::queue<int>			output_q;

// control variables
static int device_sn[iTFS::max_device];
static int device_idx[iTFS::max_device];

typedef enum {
	status_normal = 0x00,
	status_underrun = 0x01,
	status_overrun = 0x02,
	status_high_temp_warning = 0x03,
	status_low_temp_warning = 0x04,
	status_missing_rows = 0x10,
	status_timeout = 0xFF
}thread_status;

static int ilidar_thread_frame[iTFS::max_device];
static int ilidar_thread_status[iTFS::max_device];
static int ilidar_frame_status[iTFS::max_device];

static int output_thread_frame;
static int output_thread_idx;
static int output_thread_status;

typedef struct {
	int		cvt_flag;

	int		max_depth;	// [mm] maximum_distance_in_meter, UNIT CAUTION
	int		min_depth;	// [mm] minimum_distance_in_meter, UNIT CAUTION
	//float	hue_rot;	// [deg] hue_rotation_spanning_in_degree, SKIPPED
	//float	hue_shift;	// [deg] hue_shift_in_degree, SKIPPED

	float	norm_min_saturation;	// minimum_saturation
	float	h2s;				// hue_weight_to_saturation
	float	v2s;				// intensity_weight_to_saturation

	float	gamma_intensity;	// intensity_gamma
	float	norm_max_intensity;	// display_intensity_max
	float	norm_min_intensity;	// display_intensity_min

	int		max_intensity;		// 3 * 4096
}hsv_converter;

static hsv_converter ilidar_cvt;

static int read_ilidar_cvt(std::string file) {
	// Read the file
	FILE* fc = fopen(file.c_str(), "r");

	// Check the file
	if (fc == NULL) { return (-1); }

	// Read configuration
	int		cvt_flag;
	int		max_depth;	// [mm] maximum_distance_in_meter, UNIT CAUTION
	int		min_depth;	// [mm] minimum_distance_in_meter, UNIT CAUTION
	float	norm_min_saturation;	// minimum_saturation
	float	h2s;				// hue_weight_to_saturation
	float	v2s;				// intensity_weight_to_saturation
	float	gamma_intensity;	// intensity_gamma
	float	norm_max_intensity;	// display_intensity_max
	float	norm_min_intensity;	// display_intensity_min
	int		max_intensity;		// 3 * 4096

	fscanf(fc, "cvt_flag = %d\n", &cvt_flag);
	fscanf(fc, "max_depth = %d\n", &max_depth);
	fscanf(fc, "min_depth = %d\n", &min_depth);
	fscanf(fc, "norm_min_saturation = %f\n", &norm_min_saturation);
	fscanf(fc, "h2s = %f\n", &h2s);
	fscanf(fc, "v2s = %f\n", &v2s);
	fscanf(fc, "gamma_intensity = %f\n", &gamma_intensity);
	fscanf(fc, "norm_max_intensity = %f\n", &norm_max_intensity);
	fscanf(fc, "norm_min_intensity = %f\n", &norm_min_intensity);
	fscanf(fc, "max_intensity = %d\n", &max_intensity);

	ilidar_cvt.cvt_flag = cvt_flag;
	ilidar_cvt.max_depth = max_depth;
	ilidar_cvt.min_depth = min_depth;
	ilidar_cvt.norm_min_saturation = norm_min_saturation;
	ilidar_cvt.h2s = h2s;
	ilidar_cvt.v2s = v2s;
	ilidar_cvt.gamma_intensity = gamma_intensity;
	ilidar_cvt.norm_max_intensity = norm_max_intensity;
	ilidar_cvt.norm_min_intensity = norm_min_intensity;
	ilidar_cvt.max_intensity = max_intensity;

	fclose(fc);
	return (0);
}

typedef struct {
	int		ilidar_idx[iTFS::max_device];
	int		ilidar_serial_number[iTFS::max_device];
	int		ilidar_update[iTFS::max_device];
	int		ilidar_num;
	int		ilidar_sync;

	int		shm_size;

	int		display_period;
	int		print_period;

	uint8_t		dest_ip[4];
	uint16_t	dest_port;

	uint16_t	output_src_port;
	uint16_t	output_dest_port;

	int		ilidar_period;
	int		ilidar_timeout;
}multi_ilidar;

static multi_ilidar ilidar_set;

static int read_ilidar_num(std::string file) {
	// Read the file
	FILE* fc = fopen(file.c_str(), "r");

	// Check the file
	if (fc == NULL) { return (-1); }

	// Read configuration
	int	shm_size = 4;

	int lidar_num = 0;
	int ilidar_sync = 0;
	int serial_num = 0;

	int display_period = 0;
	int print_period = 0;

	int dest_ip[4];
	int dest_port;

	int output_src_port;
	int output_dest_port;

	int ilidar_period = 0;
	int ilidar_timeout = 0;

	fscanf(fc, "LN = %d\n", &lidar_num);
	fscanf(fc, "SHM_SIZE = %d\n", &shm_size);
	fscanf(fc, "SYNC = %d\n", &ilidar_sync);
	fscanf(fc, "PERIOD = %d\n", &ilidar_period);
	fscanf(fc, "TIMEOUT = %d\n", &ilidar_timeout);
	fscanf(fc, "DISPLAY = %d\n", &display_period);
	fscanf(fc, "PRINT = %d\n", &print_period);
	fscanf(fc, "DEST_IP = %d.%d.%d.%d\n", &dest_ip[0], &dest_ip[1], &dest_ip[2], &dest_ip[3]);
	fscanf(fc, "DEST_PORT = %d\n", &dest_port);
	fscanf(fc, "OUTPUT_SRC_PORT = %d\n", &output_src_port);
	fscanf(fc, "OUTPUT_DEST_PORT = %d\n", &output_dest_port);

	if (lidar_num < 1) { fclose(fc); return (-2); }
	else if (lidar_num > iTFS::max_device) { fclose(fc); return (-2); }

	ilidar_set.ilidar_num = lidar_num;
	ilidar_set.ilidar_sync = ilidar_sync;
	ilidar_set.shm_size = shm_size;

	if (display_period < 33) { display_period = 33; }
	if (print_period < 100) { print_period = 100; }

	ilidar_set.display_period = display_period;
	ilidar_set.print_period = print_period;

	ilidar_set.dest_ip[0] = dest_ip[0];
	ilidar_set.dest_ip[1] = dest_ip[1];
	ilidar_set.dest_ip[2] = dest_ip[2];
	ilidar_set.dest_ip[3] = dest_ip[3];
	ilidar_set.dest_port = dest_port;

	ilidar_set.output_src_port = output_src_port;
	ilidar_set.output_dest_port = output_dest_port;

	for (int _i = 0; _i < lidar_num; _i++) {
		fscanf(fc, "SN = %d\n", &serial_num);
		ilidar_set.ilidar_idx[_i] = -1;
		ilidar_set.ilidar_serial_number[_i] = serial_num;
		ilidar_set.ilidar_update[_i] = 0;
	}

	ilidar_set.ilidar_period = ilidar_period;
	ilidar_set.ilidar_timeout = ilidar_timeout;

	fclose(fc);
	return (0);
}

// Basic lidar data handler function
static void lidar_data_handler(iTFS::device_t *device) {
	int idx = ilidar_set.ilidar_idx[device->idx];
	if (idx < 0) { return; }

	// Print message
	//printf("[MESSAGE] iTFS::LiDAR image  | D# %d  M %d  F# %2d  %d.%d.%d.%d:%d\n",
	//	device->idx, device->data.mode, device->data.frame,
	//	device->ip[0], device->ip[1], device->ip[2], device->ip[3], device->port);

	// Deep-copy the lidar data to cv::Mat
	memcpy((void*)lidar_img_depth[idx].data,
		(const void*)device->data.img,
		sizeof(device->data.img) / 2);

	memcpy((void*)lidar_img_intensity[idx].data,
		(const void*)&(device->data.img[iTFS::max_row][0]),
		sizeof(device->data.img) / 2);

	// Copy row_frame status
	ilidar_frame_status[idx] = device->data.frame_status;

	// Notify the reception to the main thread
	std::lock_guard<std::mutex> lk(lidar_cv_mutex[idx]);
	lidar_q[idx].push(device->data.frame);
	lidar_cv[idx].notify_one();
}

// Basic lidar status packet handler function
static void status_packet_handler(iTFS::device_t* device) {
	// Print message
	//printf("[MESSAGE] iTFS::LiDAR status | D#%d mode %d frame %2d time %lld us temp %.2f from %3d.%3d.%3d.%3d:%5d\n",
	//	device->idx, device->status.capture_mode, device->status.capture_frame, get_sensor_time_in_us(&device->status), (float)(device->status.sensor_temp_core) * 0.01f,
	//	device->ip[0], device->ip[1], device->ip[2], device->ip[3], device->port);
}

// Basic lidar info packet handler function
static void info_packet_handler(iTFS::device_t* device) {
	int idx = ilidar_set.ilidar_idx[device->idx];
	if (idx != -1) { return; }

	// Check info packet version
	if (device->info.sensor_sn != 0) {
		// Print message
		printf("[ ERROR ] iTFS::LiDAR info packet was received. (lower version lidar)\n");
	}
	else if (device->info_v2.sensor_sn != 0) {
		printf("[MESSAGE] iTFS::LiDAR info_v2 packet was received.\n");
		printf("[MESSAGE] iTFS::LiDAR info_v2| D# %d  lock %d\n",
			device->idx, device->info_v2.lock);

		printf("\tSN #%d mode %d, rows %d, period %d\n",
			device->info_v2.sensor_sn,
			device->info_v2.capture_mode,
			device->info_v2.capture_row,
			device->info_v2.capture_period_us);

		printf("\tshutter [ %d, %d, %d, %d, %d ]\n",
			device->info_v2.capture_shutter[0],
			device->info_v2.capture_shutter[1],
			device->info_v2.capture_shutter[2],
			device->info_v2.capture_shutter[3],
			device->info_v2.capture_shutter[4]);

		printf("\tlimit [ %d, %d ]\n",
			device->info_v2.capture_limit[0],
			device->info_v2.capture_limit[1]);

		printf("\tip   %d.%d.%d.%d\n",
			device->info_v2.data_sensor_ip[0],
			device->info_v2.data_sensor_ip[1],
			device->info_v2.data_sensor_ip[2],
			device->info_v2.data_sensor_ip[3]);

		printf("\tdest %d.%d.%d.%d:%d\n",
			device->info_v2.data_dest_ip[0],
			device->info_v2.data_dest_ip[1],
			device->info_v2.data_dest_ip[2],
			device->info_v2.data_dest_ip[3],
			device->info_v2.data_port);

		printf("\tsync %d, syncBase %d autoReboot %d, autoRebootTick %d\n",
			device->info_v2.sync,
			device->info_v2.sync_trig_delay_us,
			device->info_v2.arb,
			device->info_v2.arb_timeout);

		printf("\tFW version: V%d.%d.%d - ",
			device->info_v2.sensor_fw_ver[2],
			device->info_v2.sensor_fw_ver[1],
			device->info_v2.sensor_fw_ver[0]);
		printf((const char*)device->info_v2.sensor_fw_time);
		printf(" ");
		printf((const char*)device->info_v2.sensor_fw_date);
		printf("\n");

		printf("\tFW0: V%d.%d.%d,  FW1: V%d.%d.%d,  FW2: V%d.%d.%d\n",
			device->info_v2.sensor_fw0_ver[2],
			device->info_v2.sensor_fw0_ver[1],
			device->info_v2.sensor_fw0_ver[0],
			device->info_v2.sensor_fw1_ver[2],
			device->info_v2.sensor_fw1_ver[1],
			device->info_v2.sensor_fw1_ver[0],
			device->info_v2.sensor_fw2_ver[2],
			device->info_v2.sensor_fw2_ver[1],
			device->info_v2.sensor_fw2_ver[0]);

		if (device->info_v2.sensor_boot_mode == 0) {
			printf("\tSENSOR IS IN SAFE-MODE\n");
		}

		// Check idx
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			if (ilidar_set.ilidar_serial_number[_i] == (int)device->info_v2.sensor_sn) {
				ilidar_set.ilidar_idx[device->idx] = _i;
				ilidar_set.ilidar_update[_i] = 1;

				// Add
				printf("[MESSAGE] iTFS::LiDAR info_v2| D#%d SN#%d : Added to L#%d\n",
					device->idx, device->info_v2.sensor_sn, _i);

				return;
			}
		}

		// Error
		printf("[MESSAGE] iTFS::LiDAR info_v2| D#%d SN#%d : NOT FOUND ON THE LIST!!!\n",
			device->idx, device->info_v2.sensor_sn);
	}
	else {
		printf("[ ERROR ] iTFS::LiDAR info   | INVALID PACKET!!!\n");
	}
}

// Example keyboard input run in seperate thread 
static void keyboard_input_run(iTFS::LiDAR* ilidar) {
	// Wait fot the sensor
	while (ilidar->Ready() != true) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }

	// Check keyboard input
	while (ilidar->Ready() == true) {
		// Get char input
		char ch = getchar();
	}
}

bool ilidar_thread_run = true;
int ilidar_thread_cvt = 0;

void lidar_img_handler(int idx) {
	ilidar_thread_status[idx] = status_normal;	// normal
	ilidar_thread_frame[idx] = 0;

	// status for condition variable
	bool wait_status;

	// frame number for overrun check
	int frame;

	// Outputs
	cv::Mat output_hsv_raw = cv::Mat::zeros(iTFS::max_row, iTFS::max_col, CV_8UC3);
	cv::Mat output_hsv_bgr = cv::Mat::zeros(iTFS::max_row, iTFS::max_col, CV_8UC3);
	cv::Mat output_hsv_roi = output_hsv(cv::Rect(0, idx * iTFS::max_row, iTFS::max_col, iTFS::max_row));
	cv::Mat output_raw_depth_roi = output_raw(cv::Rect(0, idx * iTFS::max_row, iTFS::max_col, iTFS::max_row));
	cv::Mat output_raw_intensity_roi = output_raw(cv::Rect(0, (ilidar_set.ilidar_num + idx) * iTFS::max_row, iTFS::max_col, iTFS::max_row));

	// Main loop starts here
	while (ilidar_thread_run) {
		// Wait for new data
		std::unique_lock<std::mutex> lk(lidar_cv_mutex[idx]);
		wait_status = lidar_cv[idx].wait_for(lk, std::chrono::milliseconds(500), [=] { return !lidar_q[idx].empty(); });

		// Check the wait status
		if (!wait_status) {
			// Set status to underrun
			ilidar_thread_status[idx] = status_underrun;	// warning

			/* The main loop is slower than data reception handler */
			// printf("[WARNING] iTFS::LiDAR Thread #%d does not receive the LiDAR data for 0.5 sec!\n", idx);
			continue;
		}

		// Pop the front value
		frame = lidar_q[idx].front();
		lidar_q[idx].pop();

		// Check the main loop underrun
		if (!lidar_q[idx].empty()) {
			// Set status to overrun
			ilidar_thread_status[idx] = status_overrun;	// warning

			/* The main loop is slower than data reception handler */
			// printf("[WARNING] iTFS::LiDAR Thread #%d seems to be slower than the LiDAR data reception handler.\n", idx);

			// Flush the queue
			while (!lidar_q[idx].empty()) { frame = lidar_q[idx].front(); lidar_q[idx].pop(); }
		}

		// Check frame status
		if (ilidar_frame_status[idx] != 0) { ilidar_thread_status[idx] |= status_missing_rows; }

		// Copy the frame number for monitoring
		ilidar_thread_frame[idx] = frame;

		// Check convert flag
		if (ilidar_thread_cvt) {
			/* CONVERT TO HSV IMG */
			uint16_t* depth_ptr, * intensity_ptr;
			depth_ptr = (uint16_t*)lidar_img_depth[idx].data;
			intensity_ptr = (uint16_t*)lidar_img_intensity[idx].data;
			for (int _v = 0; _v < iTFS::max_row; _v++) {
				for (int _u = 0; _u < iTFS::max_col; _u++) {
					// Get hue
					float hue = (depth_ptr[iTFS::max_col * _v + _u] > ilidar_cvt.max_depth) ? (1.0f) : ((depth_ptr[iTFS::max_col * _v + _u] < ilidar_cvt.min_depth) ? (0.0f) : ((float)(depth_ptr[iTFS::max_col * _v + _u] - ilidar_cvt.min_depth) / (float)(ilidar_cvt.max_depth - ilidar_cvt.min_depth)));

					// Get value
					float norm_intensity = pow((float)intensity_ptr[iTFS::max_col * _v + _u] / (float)ilidar_cvt.max_intensity, ilidar_cvt.gamma_intensity);
					norm_intensity = norm_intensity * (ilidar_cvt.norm_max_intensity - ilidar_cvt.norm_min_intensity) + ilidar_cvt.norm_min_intensity;
					norm_intensity = (norm_intensity > 1.0f) ? (1.0f) : (norm_intensity);
					float value = norm_intensity;

					// Get saturation
					float saturation = ilidar_cvt.h2s * hue + ilidar_cvt.v2s * value;
					saturation = (saturation > 1.0f) ? (1.0f) : (saturation);
					saturation = saturation * (1.0f - ilidar_cvt.norm_min_saturation) + ilidar_cvt.norm_min_saturation;

					// Store
					cv::Vec3b& pixel = output_hsv_raw.at<cv::Vec3b>(_v, _u);
					pixel[0] = (uchar)(180.0f * hue);
					pixel[1] = (uchar)(255.0f * saturation);
					pixel[2] = (uchar)(255.0f * value);
				}
			}

			cv::cvtColor(output_hsv_raw, output_hsv_bgr, cv::COLOR_HSV2BGR);

			auto img_time = std::chrono::system_clock::now();

			lidar_img_mutex.lock();
			output_hsv_bgr.copyTo(output_hsv_roi);
			if (ilidar_thread_cvt == 2)
			{
				lidar_img_depth[idx].copyTo(output_raw_depth_roi);
				lidar_img_intensity[idx].copyTo(output_raw_intensity_roi);
			}
			lidar_img_time[idx] = img_time;
			lidar_img_ready[idx] = true;
			lidar_img_mutex.unlock();
		}
		else {
			/* RAW IMG */
			auto img_time = std::chrono::system_clock::now();

			lidar_img_mutex.lock();
			lidar_img_depth[idx].copyTo(output_raw_depth_roi);
			lidar_img_intensity[idx].copyTo(output_raw_intensity_roi);
			lidar_img_time[idx] = img_time;
			lidar_img_ready[idx] = true;
			lidar_img_mutex.unlock();
		}

		//if (idx == (ilidar_set.ilidar_num - 1)) {
		//	// Notify to the output thread
		//	std::lock_guard<std::mutex> output_lk(output_cv_mutex);
		//	output_q.push(frame);
		//	output_cv.notify_one();
		//}
	}
}

int		shared_memory_id;
int		shared_memory_size;
void	*shared_memory_ptr = NULL;
#define	SHARED_MEMORY_BUFFER_SIZE	((int)4)

void lidar_output_run() {
	output_thread_status = status_normal;	// normal
	output_thread_frame = 0;
	output_thread_idx = 0;

	// status for condition variable
	bool wait_status;

	// frame number for overrun check
	int frame;

	// Create socket for notification
	int					local_sockfd;
	struct sockaddr_in	addr, addr_app;

	// Socker message
	uint8_t output_msg[130];
	if (ilidar_thread_cvt) { output_msg[0] = (1 << 7) | ilidar_set.ilidar_num; }
	else  { output_msg[0] = (0 << 7) | ilidar_set.ilidar_num; }

#if defined (_WIN32) || defined( _WIN64)
	// WSA Startup
	WSADATA wsa_data;
	if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != NO_ERROR) {
		printf("[ ERROR ] iTFS::LiDAR WSA Loading Failed!\n");
		return;
	}
#endif // WSA
	
	// Open the socket
	if ((local_sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
		printf("[ ERROR ] iTFS::LiDAR Local Socket Opening Failed\n");
		return;
	}

	// Initialize outgoing address
	memset((void*)&addr, 0x00, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	addr.sin_port = htons(ilidar_set.output_src_port);

	memset(&addr_app, 0, sizeof(addr_app));
	addr_app.sin_family = AF_INET;
	addr_app.sin_addr.s_addr = inet_addr("127.0.0.1");
	addr_app.sin_port = htons(ilidar_set.output_dest_port);

	int enable = 1;

	// Set the socket option to reuse address
	if (setsockopt(local_sockfd, SOL_SOCKET, SO_REUSEADDR, (char*)&enable, sizeof(enable)) < 0) {
		printf("[ ERROR ] iTFS::LiDAR Sender Socket Setsockopt Failed\n");
		closesocket(local_sockfd);
		return;
	}

	// Bind the socket
	if (bind(local_sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		printf("[ ERROR ] iTFS::LiDAR Socket Bind Failed\n");
		closesocket(local_sockfd);
		return;
	}

	// Last output time
	auto pri_time = std::chrono::system_clock::now();

	// Local status
	int lidar_output_frame = 0;
	int lidar_frame_skip_count[iTFS::max_device] = { 0, };
	int lidar_img_time_duration[iTFS::max_device];

	// Local copy
	bool lidar_img_ready_local[iTFS::max_device];
	std::chrono::time_point<std::chrono::system_clock> lidar_img_time_local[iTFS::max_device];

	// Wait for stream start
	while (!stream_ready) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	// Timeout variables
	int check_period = ilidar_set.ilidar_period + 10;
	int check_timeout = ilidar_set.ilidar_timeout;

	// Main loop starts here
	while (ilidar_thread_run) {
		// Get current time
		auto cur_time = std::chrono::system_clock::now();

		// Copy 
		lidar_img_mutex.lock();
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			lidar_img_ready_local[_i] = lidar_img_ready[_i];
			lidar_img_time_local[_i] = lidar_img_time[_i];
		}
		lidar_img_mutex.unlock();

		// Check wait or not
		bool wait = false;
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			if (!lidar_img_ready_local[_i]) {
				auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - lidar_img_time_local[_i]);
				if (duration_ms.count() < check_period) {
					// Wait
					wait = true;
					break;
				}
			}
		}

		if (wait) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// Check data arrival
		bool arrived = false;
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			if (lidar_img_ready_local[_i]) {
				arrived = true;
				break;
			}
		}

		if (!arrived) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - lidar_img_time_local[_i]);
			lidar_img_time_duration[_i] = duration_ms.count();
		}

		// Check frame skip
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			if (lidar_img_ready_local[_i]) {
				lidar_frame_skip_count[_i] = 0;
			}
			else {
				lidar_frame_skip_count[_i]++;
			}

			if (lidar_img_time_duration[_i] > check_timeout) {
				printf("L#%d: TIME OUT!!!\n", _i);
				ilidar_thread_status[_i] = status_timeout;
			}
		}

		lidar_img_mutex.lock();
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			lidar_img_ready[_i] = false;
		}
		lidar_img_mutex.unlock();

		lidar_output_frame++;

		// // Wait for new output
		// std::unique_lock<std::mutex> output_lk(output_cv_mutex);
		// wait_status = output_cv.wait_for(output_lk, std::chrono::milliseconds(500), [=] { return !output_q.empty(); });

		// // Check the wait status
		// if (!wait_status) {
		// 	// Set status to underrun
		// 	output_thread_status = status_underrun;	// warning
		// 	continue;
		// }

		// // Pop the front value
		// frame = output_q.front();
		// output_q.pop();

		// // Check the main loop underrun
		// if (!output_q.empty()) {
		// 	// Set status to overrun
		// 	output_thread_status = status_overrun;	// warning

		// 	// Flush the queue
		// 	while (!output_q.empty()) { frame = output_q.front(); output_q.pop(); }
		// }

		// // Update frame
		// output_thread_frame = frame;

		// Copy the output data in shared memory
		lidar_img_mutex.lock();
		if (ilidar_thread_cvt == 1) {
			void *target_ptr = (void*)((uint8_t*)shared_memory_ptr + output_thread_idx * (4 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num));
			memcpy(target_ptr,
				(const void*)output_hsv.data,
				3 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num);
		}
		else if (ilidar_thread_cvt == 2)
		{
			void *target_ptr_hsv = (void*)((uint8_t*)shared_memory_ptr + output_thread_idx * (7 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num));
			void *target_ptr_raw = (void*)((uint8_t*)shared_memory_ptr + output_thread_idx * (7 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num) + (3 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num));
			
			
			memcpy(target_ptr_hsv,
				(const void*)output_hsv.data,
				3 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num);
			memcpy(target_ptr_raw,
			(const void*)output_raw.data,
			4 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num);
		}
		else {
			void *target_ptr = (void*)((uint8_t*)shared_memory_ptr + output_thread_idx * (4 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num));
			memcpy(target_ptr,
				(const void*)output_raw.data,
				4 * sizeof(uint8_t) * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num);
		}
		lidar_img_mutex.unlock();

		// Store output message
		output_msg[1] = output_thread_idx;
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			output_msg[4*_i + 2] = (device_sn[_i] >> 0) & 0xFF;
			output_msg[4*_i + 3] = (device_sn[_i] >> 8) & 0xFF;
			output_msg[4*_i + 4] = ilidar_thread_frame[_i];
			output_msg[4*_i + 5] = ilidar_thread_status[_i];
		}

		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			ilidar_thread_status[_i] = status_normal;
		}

		// Send udp packet to application
		if (sendto(local_sockfd, (const char*)output_msg, sizeof(output_msg), 0, (struct sockaddr*)&addr_app, sizeof(addr_app)) < 0) {
			output_thread_status = status_timeout;
		}
		
		if (++output_thread_idx >= ilidar_set.shm_size) { output_thread_idx = 0; }
	}
}

// Color text functions
static inline void reset_printf(void) { printf("\033[0m");}
static inline void set_printf_bold(void) { printf("\033[1;1m");}
static inline void set_printf_underline(void) { printf("\033[1;4m");}
static inline void set_printf_invert(void) { printf("\033[1;7m");}
static inline void set_printf_bold_off(void) { printf("\033[1;21m");}
static inline void set_printf_underline_off(void) { printf("\033[1;24m");}
static inline void set_printf_invert_off(void) { printf("\033[1;27m");}

static inline void set_printf_color_K(void) { printf("\033[1;30m");}
static inline void set_printf_color_R(void) { printf("\033[1;31m");}
static inline void set_printf_color_G(void) { printf("\033[1;32m");}
static inline void set_printf_color_Y(void) { printf("\033[1;33m");}
static inline void set_printf_color_B(void) { printf("\033[1;34m");}
static inline void set_printf_color_M(void) { printf("\033[1;35m");}
static inline void set_printf_color_C(void) { printf("\033[1;36m");}
static inline void set_printf_color_W(void) { printf("\033[1;37m");}

static inline void set_printf_background_K(void) { printf("\033[1;40m");}
static inline void set_printf_background_R(void) { printf("\033[1;41m");}
static inline void set_printf_background_G(void) { printf("\033[1;42m");}
static inline void set_printf_background_Y(void) { printf("\033[1;43m");}
static inline void set_printf_background_B(void) { printf("\033[1;44m");}
static inline void set_printf_background_M(void) { printf("\033[1;45m");}
static inline void set_printf_background_C(void) { printf("\033[1;46m");}
static inline void set_printf_background_W(void) { printf("\033[1;47m");}

static void print_ascii_art(void) {
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("================================================================================");
	reset_printf(); printf("\n");
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("         _/  _/      _/  _/_/      _/    _/_/          _/_/_/       _/_/_/      ");
	reset_printf(); printf("\n");
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("            _/          _/  _/  _/  _/  _/  _/          _/    _/   _/           ");
	reset_printf(); printf("\n");
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("       _/  _/      _/  _/  _/  _/_/_/  _/_/    _/_/_/  _/  _/  _/ _/_/_/        ");
	reset_printf(); printf("\n");
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("      _/  _/      _/  _/  _/  _/  _/  _/  _/          _/  _/  _/ _/             ");
	reset_printf(); printf("\n");
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("     _/  _/_/_/  _/  _/_/    _/  _/  _/  _/          _/    _/   _/              ");
	reset_printf(); printf("\n");
	set_printf_bold(); set_printf_color_W(); set_printf_background_K();
	printf("================================================================================");
	reset_printf(); printf("\n");
}

void display_imshow_run(void) {
	cv::namedWindow("DISPLAY", cv::WINDOW_NORMAL);
	cv::waitKey(ilidar_set.display_period);

	while (ilidar_thread_run) {
		// Call imshow for monitoring
		lidar_img_mutex.lock();
		if (ilidar_thread_cvt) { cv::imshow("DISPLAY", output_hsv); }
		else { cv::imshow("DISPLAY", output_raw); }
		lidar_img_mutex.unlock();
		cv::waitKey(ilidar_set.display_period);
	}
}

// Helloworld example starts here
int main(int argc, char* argv[]) {
	// Get program start time
	auto pri_time = std::chrono::system_clock::now();

	// ASCII ART!!
	print_ascii_art();

	// Print version
	printf("[MESSAGE] iTFS::LiDAR This is multi_thread_read process ");
	set_printf_bold();
	set_printf_background_K();
	printf("( V %d.%d.%d)", version[2], version[1], version[0]);
	reset_printf();
	printf("\n");

	// Check lib version
	if (iTFS::ilidar_lib_ver[2] < 1 || iTFS::ilidar_lib_ver[1] < 12) {
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("[ ERROR ] iTFS::LiDAR LIB version is not macthed !!");
		reset_printf();
		printf("\n");
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("\tRequired ( V 1.12.0+ ), but current ( V %d.%d.%d )", iTFS::ilidar_lib_ver[2], iTFS::ilidar_lib_ver[1], iTFS::ilidar_lib_ver[0]);
		reset_printf();
		printf("\n");

		exit(-1);
	}

	// Paramter files
	std::string ilidar_num_file_name, ilidar_cvt_file_name;

	// Display flags
	bool display_imshow = false, print_info = false;
	
	// Check the input arguments
	if (argc > 0) {
		for (int _i = 0; _i < argc; _i++) {
			// Check the ilidar_num file name
			if (std::strcmp(argv[_i], "-i") == 0 && ((_i + 1) < argc)) {
				/* Success to get ilidar_num file name */
				ilidar_num_file_name = argv[_i + 1];
			}

			// Check the ilidar_cvt file name
			if (std::strcmp(argv[_i], "-c") == 0 && ((_i + 1) < argc)) {
				/* Success to get ilidar_cvt file name */
				ilidar_cvt_file_name = argv[_i + 1];
			}

			// Check the display flag
			if (std::strcmp(argv[_i], "-d") == 0) {
				/* Set display imshow flag to true */
				display_imshow = true;
			}

			// Check the print flag
			if (std::strcmp(argv[_i], "-p") == 0) {
				/* Set print info flag to true */
				print_info = true;
			}
		}
	}

	// Check the file names
	if (ilidar_num_file_name.empty() || ilidar_cvt_file_name.empty()) {
		ilidar_num_file_name = "ilidar_cfg.txt";
		ilidar_cvt_file_name = "ilidar_cvt.txt";
	}

	// Read ilidar_num file
	int read_ilidar = read_ilidar_num(ilidar_num_file_name);
	if (read_ilidar == (-1)) {
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("[ ERROR ] iTFS::LiDAR There is no file for ilidar serial number (%s)", ilidar_num_file_name.c_str());
		reset_printf();
		printf("\n");
		return 0;
	}
	else if (read_ilidar == (-2)) {
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("[ ERROR ] iTFS::LiDAR Invalid ilidar serial number in (%s)", ilidar_num_file_name.c_str());
		reset_printf();
		printf("\n");
		return 0;
	}
	else {
		printf("[MESSAGE] iTFS::LiDAR From the ilidar serial number file (%s),\n", ilidar_num_file_name.c_str());
		printf("\tThis program receive the data from %d LiDARs,\n\t[", ilidar_set.ilidar_num);
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			printf(" %d ", ilidar_set.ilidar_serial_number[_i]);
		}
		printf("]\n");
		printf("\tSynchronization period = %d sec\n", ilidar_set.ilidar_sync);

		printf("\tDisplay period = %d msec\n", ilidar_set.display_period);
		printf("\tPrint period = %d msec\n", ilidar_set.print_period);

		set_printf_bold();
		set_printf_background_K();
		printf("\t PC IP : %d.%d.%d.%d   PORT: %d",
			ilidar_set.dest_ip[0], ilidar_set.dest_ip[1], ilidar_set.dest_ip[2], ilidar_set.dest_ip[3], ilidar_set.dest_port);
		reset_printf();
		printf("\n");

		set_printf_bold();
		set_printf_background_K();
		printf("\tOUTPUT : 127.0.0.1  [%d] --> [%d]",
			ilidar_set.output_src_port, ilidar_set.output_dest_port);
		reset_printf();
		printf("\n");
	}

	// Read ilidar_cvt file
	int read_cvt = read_ilidar_cvt(ilidar_cvt_file_name);
	if (read_cvt == (-1)) {
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("[ ERROR ] iTFS::LiDAR There is no file for ilidar converter (%s)", ilidar_cvt_file_name.c_str());
		reset_printf();
		printf("\n");
		return 0;
	}
	else {
		printf("[MESSAGE] iTFS::LiDAR From the ilidar converter file (%s),\n", ilidar_cvt_file_name.c_str());

		if (ilidar_cvt.cvt_flag == 0) {
			ilidar_thread_cvt = ilidar_cvt.cvt_flag;
			set_printf_bold();
			set_printf_background_K();
			printf("\tcvt_flag = 0, RAW MODE ON");
			reset_printf();
			printf("\n");
		}
		else if(ilidar_cvt.cvt_flag == 2)
		{
			ilidar_thread_cvt = ilidar_cvt.cvt_flag;
			set_printf_bold();
			set_printf_background_K();
			printf("\tcvt_flag = 2, [");
			set_printf_color_R();
			printf("H");
			set_printf_color_G();
			printf("S");
			set_printf_color_B();
			printf("V");
			reset_printf();
			set_printf_bold();
			set_printf_background_K();
			printf(" CONVERTION ON]");
			reset_printf();
			printf("\n");
			printf("\tThis program convet the data with\n");
			printf("\tmax_depth = %d\n", ilidar_cvt.max_depth);
			printf("\tmin_depth = %d\n", ilidar_cvt.min_depth);
			printf("\tnorm_min_saturation = %f\n", ilidar_cvt.norm_min_saturation);
			printf("\th2s = %f\n", ilidar_cvt.h2s);
			printf("\tv2s = %f\n", ilidar_cvt.v2s);
			printf("\tgamma_intensity = %f\n", ilidar_cvt.gamma_intensity);
			printf("\tnorm_max_intensity = %f\n", ilidar_cvt.norm_max_intensity);
			printf("\tnorm_min_intensity = %f\n", ilidar_cvt.norm_min_intensity);
			printf("\tmax_intensity = %d\n", ilidar_cvt.max_intensity);
		}
		else {
			ilidar_thread_cvt = ilidar_cvt.cvt_flag;
			set_printf_bold();
			set_printf_background_K();
			printf("\tcvt_flag = 1, [");
			set_printf_color_R();
			printf("H");
			set_printf_color_G();
			printf("S");
			set_printf_color_B();
			printf("V");
			reset_printf();
			set_printf_bold();
			set_printf_background_K();
			printf(" CONVERTION ON]");
			reset_printf();
			printf("\n");
			printf("\tThis program convet the data with\n");
			printf("\tmax_depth = %d\n", ilidar_cvt.max_depth);
			printf("\tmin_depth = %d\n", ilidar_cvt.min_depth);
			printf("\tnorm_min_saturation = %f\n", ilidar_cvt.norm_min_saturation);
			printf("\th2s = %f\n", ilidar_cvt.h2s);
			printf("\tv2s = %f\n", ilidar_cvt.v2s);
			printf("\tgamma_intensity = %f\n", ilidar_cvt.gamma_intensity);
			printf("\tnorm_max_intensity = %f\n", ilidar_cvt.norm_max_intensity);
			printf("\tnorm_min_intensity = %f\n", ilidar_cvt.norm_min_intensity);
			printf("\tmax_intensity = %d\n", ilidar_cvt.max_intensity);
		}
	}

	lidar_output_raw = cv::Mat::zeros(ilidar_set.ilidar_num * 2 * iTFS::max_row, iTFS::max_col, CV_16UC1);

	output_raw = cv::Mat::zeros(ilidar_set.ilidar_num * 2 * iTFS::max_row, iTFS::max_col, CV_16UC1);
	output_hsv = cv::Mat::zeros(ilidar_set.ilidar_num * iTFS::max_row, iTFS::max_col, CV_8UC3);

	for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
		lidar_img_depth[_i] = lidar_output_raw(cv::Rect(0, _i * iTFS::max_row, iTFS::max_col, iTFS::max_row));
		lidar_img_intensity[_i] = lidar_output_raw(cv::Rect(0, (ilidar_set.ilidar_num + _i) * iTFS::max_row, iTFS::max_col, iTFS::max_row));
	}

	// Create threads
	ilidar_thread_run = true;
	std::thread output_thread = std::thread([=] { lidar_output_run(); });
	std::thread ilidar_thread[iTFS::max_device];
	for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
		ilidar_thread[_i] = std::thread([=] { lidar_img_handler(_i); });
	}
	printf("[MESSAGE] iTFS::LiDAR success to create %d threads.\n", ilidar_set.ilidar_num);

	// Check display flag
	if (display_imshow) {
		printf("[MESSAGE] iTFS::LiDAR ");
		set_printf_bold();
		set_printf_background_K();
		printf("imshow display ON");
		reset_printf();
		printf("\n");
	}
	else {
		printf("[MESSAGE] iTFS::LiDAR ");
		set_printf_bold();
		set_printf_background_K();
		printf("imshow display OFF");
		reset_printf();
		printf("\n");
	}

	// Check print flag
	if (print_info) {
		printf("[MESSAGE] iTFS::LiDAR ");
		set_printf_bold();
		set_printf_background_K();
		printf("print info ON");
		reset_printf();
		printf("\n");
	}
	else {
		printf("[MESSAGE] iTFS::LiDAR ");
		set_printf_bold();
		set_printf_background_K();
		printf("print info OFF");
		reset_printf();
		printf("\n");
	}

	// Create shared memory
#if defined(_WIN32) || defined( _WIN64)
	// Windows
	if(ilidar_cvt.cvt_flag != 2)
		shared_memory_size = ilidar_set.shm_size * sizeof(uint8_t) * 4 * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num;
	else
		shared_memory_size = ilidar_set.shm_size * sizeof(uint8_t) * 7 * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num;
	
	printf("[MESSAGE] iTFS::LiDAR Try to create new ");
	printf("shared memory %d KB (%d buffers) with key %d", shared_memory_size / 1024, ilidar_set.shm_size, ilidar_set.output_src_port);
	printf("\n\tThis process will delete all non-attached shared memory before create new one\n");

	HANDLE	shm_handle;
	char	shm_name[17] = "iLidar_shm_00000";
	shm_name[11] = shm_name[11] + (ilidar_set.output_src_port % 100000) / 10000;
	shm_name[12] = shm_name[12] + (ilidar_set.output_src_port % 10000) / 1000;
	shm_name[13] = shm_name[13] + (ilidar_set.output_src_port % 1000) / 100;
	shm_name[14] = shm_name[14] + (ilidar_set.output_src_port % 100) / 10;
	shm_name[15] = shm_name[15] + ilidar_set.output_src_port % 10;

	shm_handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, shared_memory_size, shm_name);

	if (shm_handle == NULL) {
		printf("Could not create file mapping object (%d).\n", GetLastError());
		return 0;
	}

	shared_memory_ptr = (void*)MapViewOfFile(shm_handle, FILE_MAP_ALL_ACCESS, 0, 0, shared_memory_size);

	if (shared_memory_ptr == NULL) {
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(shm_handle);
		return 0;
	}

	CopyMemory(shared_memory_ptr, TEXT("shm_test"), (_tcslen(TEXT("shm_test")) * sizeof(TCHAR)));

#else
	if(ilidar_cvt.cvt_flag != 2)
		shared_memory_size = ilidar_set.shm_size * sizeof(uint8_t) * 4 * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num;
	else
		shared_memory_size = ilidar_set.shm_size * sizeof(uint8_t) * 7 * iTFS::max_row * iTFS::max_col * ilidar_set.ilidar_num;

	printf("[MESSAGE] iTFS::LiDAR Try to create new ");
	set_printf_bold();
	set_printf_background_K();
	printf("shared memory %d KB (%d buffers) with key %d", shared_memory_size/1024, ilidar_set.shm_size, ilidar_set.output_src_port);
	reset_printf();
	printf("\n\tThis process will delete all non-attached shared memory before create new one\n", shared_memory_size/1024, ilidar_set.output_src_port);
	struct shmid_ds shm_info, shm_seg;
	int max_shmid = shmctl(0, SHM_INFO, &shm_info);
	if (max_shmid > 0) {
		for (int _i = 0; _i < max_shmid; _i++) {
			int shmid = shmctl(_i, SHM_STAT, &shm_seg);
			if (shmid <= 0) {
				continue;
			}
			else if (shm_seg.shm_nattch == 0) {
				if ((shmctl(shmid, IPC_RMID, 0)) == -1) {
					set_printf_bold();
					set_printf_color_R();
					set_printf_background_K();
					printf("[ ERROR ] iTFS::LiDAR Fail to remove shared memory pointer with key %d", ilidar_set.output_src_port);
					reset_printf();
					printf("\n");
					return 0;
				}
			}
		}
	}
	if ((shared_memory_id = shmget(ilidar_set.output_src_port, shared_memory_size, IPC_CREAT|0666)) == -1) {
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("[ ERROR ] iTFS::LiDAR Fail to create shared memory with key %d", ilidar_set.output_src_port);
		reset_printf();
		printf("\n");
		return 0;
	}
	if ((shared_memory_ptr = shmat(shared_memory_id, NULL, 0)) == (void *)-1) {
		set_printf_bold();
		set_printf_color_R();
		set_printf_background_K();
		printf("[ ERROR ] iTFS::LiDAR Fail to get shared memory pointer with key %d", ilidar_set.output_src_port);
		reset_printf();
		printf("\n");
		return 0;
	}
	printf("[MESSAGE] iTFS::LiDAR Success to link to the shared memory %d KB with key %d\n", shared_memory_size/1024, ilidar_set.output_src_port);
#endif
	// Delay
	printf("[MESSAGE] iTFS::LiDAR process starts in\n");
	for (int _i = 3; _i > 0; _i--) {
		printf("\t%d..\n", _i);
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	// Create iTFS LiDAR class
	iTFS::LiDAR* ilidar;
	uint8_t broadcast_ip[4] = { ilidar_set.dest_ip[0], ilidar_set.dest_ip[1], ilidar_set.dest_ip[2], 255 };
	ilidar = new iTFS::LiDAR(
		lidar_data_handler,
		status_packet_handler,
		info_packet_handler,
		broadcast_ip,
		ilidar_set.dest_ip,
		ilidar_set.dest_port);

	// Check the sensor driver is ready
	while (ilidar->Ready() != true) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
	printf("[MESSAGE] iTFS::LiDAR is ready ");
	set_printf_bold();
	set_printf_background_K();
	printf("(LIB V %d.%d.%d )", iTFS::ilidar_lib_ver[2], iTFS::ilidar_lib_ver[1], iTFS::ilidar_lib_ver[0]);
	reset_printf();
	printf("\n");

	// Create keyboard input thread
	std::thread keyboard_input_thread = std::thread([=] { keyboard_input_run(ilidar); });

	// Sync packet period
	int sync_packet_period = ilidar_set.ilidar_sync;	// [s]

	// Send first sync packet
	iTFS::packet::cmd_t sync = { 0, };
	sync.cmd_id = iTFS::packet::cmd_sync;
	sync.cmd_msg = 0;
	if (sync_packet_period != 0) {
		ilidar->Send_cmd_to_all(&sync);
		printf("[MESSAGE] iTFS::LiDAR cmd_sync packet was sent.\n");
	}
	else {
		printf("[MESSAGE] iTFS::LiDAR cmd_sync packet sender was disabled. (SYNC = 0)\n");
	}

	// Call imshow for monitoring
	std::thread display_imshow_thread;
	if (display_imshow) {
		display_imshow_thread = std::thread([=] { display_imshow_run(); });
	}

	// First sync
	while (true) {
		// Sleep
		std::this_thread::sleep_for(std::chrono::milliseconds(ilidar_set.print_period));

		// Check the update status
		int cnt = 0;
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) { cnt += ilidar_set.ilidar_update[_i]; }
		if (cnt != ilidar_set.ilidar_num) {
			printf("[MESSAGE] iTFS::LiDAR wait for the initial connnection [%d/%d]...\n", cnt, ilidar_set.ilidar_num);
			continue;
		}

		// Send sync command packet
		if (sync_packet_period != 0) {
			ilidar->Send_cmd_to_all(&sync);
			printf("[MESSAGE] iTFS::LiDAR cmd_sync packet was sent (first_sync).\n");
		}
		break;
	}

	// Get index array
	for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
		for (int _j = 0; _j < ilidar_set.ilidar_num; _j++) {
			if (ilidar_set.ilidar_idx[_j] == _i) {
				device_idx[_i] = _j;
				device_sn[_i] = ilidar->device[_j].info_v2.sensor_sn;
			}
		}
	}

	// Reset thread status
	for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
		ilidar_thread_status[_i] = status_normal;
	}

	// Start now
	stream_ready = true;

	// Main loop starts here
	while (true) {
		// Print ilidar info
		if (print_info) {
			std::time_t print_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			printf("-------------------------------------------------------------------------\n");
			printf(" LiDAR | S/N | FN | TIME                | RECV         | CORE   | CASE   \n");
			printf("-------------------------------------------------------------------------\n");
			for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
				uint16_t	sensor_sn = ilidar->device[device_idx[_i]].info_v2.sensor_sn;
				uint64_t	sensor_time_th = ilidar->device[device_idx[_i]].status_full.sensor_time_th;
				uint16_t	sensor_time_tl = ilidar->device[device_idx[_i]].status_full.sensor_time_tl;;
				float		sensor_temp_core = (float)(ilidar->device[device_idx[_i]].status_full.sensor_temp_core) * 0.01f;
				float		sensor_temp_enclosure = (float)(ilidar->device[device_idx[_i]].status_full.sensor_temp[0]) * 0.01f;
				float		sensor_temp_rx = (float)(ilidar->device[device_idx[_i]].status_full.sensor_temp_rx) * 0.01f;

				int			day = sensor_time_th / (24 * 60 * 60 * 1000);
				uint64_t	day_time = sensor_time_th % (24 * 60 * 60 * 1000);
				int			day_hour = day_time / (60 * 60 * 1000);
				int			day_min = (day_time / (60 * 1000)) % 60;
				int			day_sec = (day_time / (1000)) % 60;
				int			day_ms = day_time % 1000;

				std::time_t	recv_time = std::chrono::system_clock::to_time_t(lidar_img_time[_i]);
				std::tm		now_tm = *std::localtime(&recv_time);

				auto duration_since_epoch = lidar_img_time[_i].time_since_epoch();
				auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch) % 1000;

				printf("   %02d  | %03d | %02d | D+%3d  %02d:%02d:%02d.%03d | %02d:%02d:%02d.%03d | %04.01f ℃ | %04.01f ℃\n",
					_i, sensor_sn, ilidar_thread_frame[_i],
					day,
					day_hour, day_min, day_sec, day_ms,
					now_tm.tm_hour, now_tm.tm_min, now_tm.tm_sec, millis.count(),
					sensor_temp_core, sensor_temp_enclosure);
			}
			printf("-------------------------------------------------------------------------\n");
			printf("  OUTPUT  |  LiDAR %02d  |  FRAME %02d  |  INDEX %d\n",
				(ilidar_set.ilidar_num - 1), output_thread_frame, output_thread_idx);
		}

		// Print ilidar status
		for (int _i = 0; _i < ilidar_set.ilidar_num; _i++) {
			if (ilidar_thread_status[device_idx[_i]] == status_underrun) {
				set_printf_bold(); set_printf_color_R(); set_printf_background_K();
				printf("[WARNING] iTFS::LiDAR Thread #%d does not receive the LiDAR data for 0.5 sec!", device_idx[_i]);
				reset_printf(); printf("\n");
				ilidar_thread_status[device_idx[_i]] = status_normal;
			}
			else if (ilidar_thread_status[device_idx[_i]] == status_overrun) {
				set_printf_bold(); set_printf_color_R(); set_printf_background_K();
				printf("[WARNING] iTFS::LiDAR Thread #%d seems to be slower than the LiDAR data reception handler.", device_idx[_i]);
				reset_printf(); printf("\n");
				ilidar_thread_status[device_idx[_i]] = status_normal;
			}
		}

		// Print output status
		if (output_thread_status == status_underrun) {
			set_printf_bold(); set_printf_color_R(); set_printf_background_K();
			printf("[WARNING] iTFS::LiDAR Output thread does not receive the signal for 0.5 sec!");
			reset_printf(); printf("\n");
			output_thread_status = status_normal;
		}
		else if (output_thread_status == status_overrun) {
			set_printf_bold(); set_printf_color_R(); set_printf_background_K();
			printf("[WARNING] iTFS::LiDAR Output thread seems to be slower than the LiDAR data reception handler.");
			reset_printf(); printf("\n");
			output_thread_status = status_normal;
		}
		else if (output_thread_status == status_timeout) {
			set_printf_bold(); set_printf_color_R(); set_printf_background_K();
			printf("[WARNING] iTFS::LiDAR Output thread fails to send the output message!");
			reset_printf(); printf("\n");
			output_thread_status = status_normal;
		}

		// Check the time for sync packet
		if (sync_packet_period != 0) {
			auto cur_time = std::chrono::system_clock::now();
			auto after_last_sync = std::chrono::duration_cast<std::chrono::seconds>(cur_time - pri_time);
			if (after_last_sync.count() > sync_packet_period) {
				/* Time to send sync packet */
				pri_time = cur_time;

				// Send sync command packet
				ilidar->Send_cmd_to_all(&sync);
				printf("[MESSAGE] iTFS::LiDAR cmd_sync packet was sent.\n");
			}
		}

		// Sleep
		std::this_thread::sleep_for(std::chrono::milliseconds(ilidar_set.print_period));
	}

	// Stop and delete iTFS LiDAR class
	delete ilidar;
	printf("[MESSAGE] iTFS::LiDAR has been deleted.\n");

	// Wait for keyboard input thread
	keyboard_input_thread.join();

	return 0;
}
