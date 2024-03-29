# Target-less Camera-LiDAR Calibration

## Description
Targetless Calibration between Camera and LiDAR using

1. Manually selected correspondence
2. Automated correspondence detection combining [Gaussian Mixture Model (GMM)](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21893), Chamfer Matching and [Mutual Information](https://www.mendeley.com/catalogue/13a78ff7-a5cb-31e9-81d6-a6893c303e52/)

Implementation assumes a KITTI dataset folder structure like the following:
```
dataset
├── calibration
│   ├── calib_cam_to_cam.txt
│   ├── calib_imu_to_velo.txt
│   └── calib_velo_to_cam.txt
├── image_00
│   └── data
│       ├── 0000000001.png
│       └── ...
└── velodyne_points
    └── data
        ├── 0000000001.bin
        └── ...
```

## Requirement
We recommend using pipenv and the Pipfile.lock file to install necessary dependencies in a virtual environment. 
```
pipenv install --ignore-pipfile
pipenv shell
```

## Usage
```
python calibrate.py --dir <path to kitti directory> --calibration_method <either "automatic" or "manual">
python calibrate.py -d <path to kitti directory> -cm <either "automatic" or "manual">
```
Example:
```
python calibrate.py --dir data/0928-KITTI-dataset/ --calibration_method automatic
```
For manual mode, click the feature in camera image and intensity image, confirm the selection with `y`, and finish selection with `q`

The calibration result is then displayed in the consol after the algorithm has converged. Example:
```
### Optimization completed ###
xyz (m): [ 1.20684236 -1.21968689  1.20059145]
Rotation Vector: [-0.02272116 -0.07837192 -0.34919117]
```
