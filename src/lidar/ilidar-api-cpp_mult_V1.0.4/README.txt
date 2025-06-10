This file describes how to use the multi thread read C++ program.

Please, check the below to use the project.
============================================================================================================
1. Install requirements
   This project will be compiled with Cmake.
   The project uses openCV (opencv2/opencv.hpp), shared memory (sys/ipc.h, sys/shm.h).
   The python script also uses numpy, opencv and sysv_ipc.
   	1-0. Upgrade system
   		$ sudo apt-get update
   		$ sudo apt-get upgrade
   		
	1-1. Install cmake
		$ sudo apt-get install build-essential cmake
		
	1-2. Install opencv
		$ sudo apt-get install libopencv-dev python3-opencv
		
	1-3. Install pip3
		$ sudo apt-get install python3-pip
		
	1-4. Install sysv_ipc
		$ sudo pip3 install sysv_ipc
		
2. Build the project using Cmake. On the same directory of this README.txt file: 
	$ cmake .
	$ cmake --build . --config Release
	
3. Edit the configuration files in /bin folder
	3-1. Edit cfg.txt file based on your application
		LN = Number of the LiDARs
		SHM_SIZE = Size of the shared memory buffer, default = 4
		SYNC = [sec] Sync packet sending period in sec, default = 20 sec
		PERIOD = [msec] Frame drop checking period, default = 80 msec
		TIMEOUT = [msec] Timeout duration, default = 500 msec
		DISPLAY = [msec] Display period in msec using imshow function, default = 33 msec
		PRINT = [msec] Print period in msec using printf function, default = 100 msec
		DEST_IP = Destination IP of the sensors, use the IP of this system
		DEST_PORT = Destination port of the sensors 
		OUTPUT_SRC_PORT = local-loopback source port for IPC message
		OUTPUT_DEST_PORT = local-loopback destination port
		SN = Serial number of the connected sensors, The order is preserved in the stitched image.
		
	3-2. Edit cvt.txt file for HSV conversion
		cvt_flag = Conversion flag, 0 = raw mode, 1 = HSV mode
		max_depth = [mm] maximum_distance_in_meter, UNIT CAUTION
		min_depth = [mm] minimum_distance_in_meter, UNIT CAUTION
		norm_min_saturation = minimum_saturation
		h2s = hue_weight_to_saturation
		v2s = intensity_weight_to_saturation
		gamma_intensity = intensity_gamma
		norm_max_intensity = display_intensity_max
		norm_min_intensity = display_intensity_min
		max_intensity = default 12288
		
4. Run the program:
	$ cd /bin
	$ ./multi_thread_read_cmake -i (cfg file name) -c (cvt file name) (-d) (-p)
	
	-d = Use imshow display option for debug
	-p = Use printf option for monitoring	
------------------------------------------------------------------------------------------------------------
- This project use shared memory for fast data handling between C++ process and python script.
- The shared memory may be need to be deleted before starting the program.
- Monitor the status of the shared memory using:
	$ ipcs -m
============================================================================================================

Pleas, check also the example screenshots in 'pic' folder


