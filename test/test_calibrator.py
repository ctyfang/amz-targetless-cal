from calibrator import *

input_dir = '/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync/'
calib_dir = '/media/carter/Samsung_T5/3dv/2011_09_26/calibration'

calibrator = Calibrator(calib_dir, input_dir, 29)
calibrator.pc_detect(visualize=True)
print('hi')