import numpy as np
import math

def camera_intrinsic_transform(fov_x=45,fov_y=60,pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,4))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(fov_x/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(fov_y/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics
 
 
def camera_intrinsic_fov(intrinsic):
    #计算FOV
    w, h = intrinsic[0][2]*2, intrinsic[1][2]*2
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    # Go
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
    return fov_x, fov_y

intrinsic = [[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]]
print(camera_intrinsic_fov(intrinsic))