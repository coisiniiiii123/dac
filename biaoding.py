import cv2
import numpy as np
import glob
import os

# 标定棋盘格的内角点数（行数和列数）
chessboard_rows = 5
chessboard_cols = 8

# 棋盘格每格的实际尺寸（单位：毫米）
square_size = 25.0  # 每格尺寸

# 准备3D点 (0,0,0), (1,0,0), (2,0,0) ...,(chessboard_cols-1,chessboard_rows-1,0)
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2) * square_size

# 用于存储所有图像的3D点和2D点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 读取标定图片
images = glob.glob('/home/jack/YOLO_World/biaoding/*.jpg')  # 标定图片路径

if not images:
    print("未找到标定图片，请确保路径正确并包含棋盘格图片。")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格的角点
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

    # 如果找到足够的角点，则添加到点集中
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 绘制角点并显示
        img = cv2.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 输出内参矩阵和畸变系数
print("内参矩阵 (Camera Matrix):")
print(mtx)
print("\n畸变系数 (Distortion Coefficients):")
print(dist)

# 保存标定结果
# np.savez('biaoding/camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# print("\n标定结果已保存到 'camera_calibration.npz' 文件中。")