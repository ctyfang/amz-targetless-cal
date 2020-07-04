# Target-less Camera-LiDAR Calibration

## Overview
Target-less Calibration between Camera and LiDAR using

1. Manually selected correspondence
2. Automated correspondence detection combining [Gaussian Mixture Model (GMM)](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21893) and [Mutual Information](https://www.mendeley.com/catalogue/13a78ff7-a5cb-31e9-81d6-a6893c303e52/)
   
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
