# Target-less Camera-LiDAR Calibration

## Description
Target-less Calibration between Camera and LiDAR using

1. Manually selected correspondence
2. Automated correspondence detection combining [Gaussian Mixture Model (GMM)](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21893), Chamfer Matching and [Mutual Information](https://www.mendeley.com/catalogue/13a78ff7-a5cb-31e9-81d6-a6893c303e52/)

Implementation assumes a KITTI dataset folder structure like the following:

my_dir  
--images_00  
----data  
------0000000000.jpg  
--velodyne_points   
----data  
------0000000000.bin  
--calibration  
----calib_cam_to_cam.txt  
----calib_velo_to_cam.txt  
----calib_imu_to_velo.txt   

## Requirement
- opencv-contrib-python = 3.4.0.12
- opencv-python = 3.4.0.12
- pyquaternion = 0.9.5
- PrettyTable = 0.7.2
- matplotlib = 3.2.1
- open3d = 0.9.0
- numpy = 1.18.3
- KDEpy = 1.0.5
- sklearn = 0.0
- scipy = 1.4.1

## Usage
```
python3 calibrate.py
```
