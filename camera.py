import cv2
import datetime
import os

# 打开默认摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

CAPTURE_INTERVAL = 3  # 拍照间隔时间
last_capture_time = datetime.datetime.now()
num = 1

while True:
    # 读取摄像头画面
    ret, frame = cap.read()
    
    if not ret:
        print("无法获取画面")
        break

    # 显示实时画面
    cv2.imshow('Camera', frame)
    
    # 检测按键
    key = cv2.waitKey(1)
    
    # 空格键拍照
    if key == 32:
        # 生成时间戳文件名
        filename = os.path.join("/home/jack/图片", datetime.datetime.now().strftime("%Y%m%d_%H%M%S.jpg"))
        cv2.imwrite(filename, frame)
        print(f"已保存: {filename}")
    
    #  # 定时拍照
    # current_time = datetime.datetime.now()
    # if (current_time - last_capture_time).seconds >= CAPTURE_INTERVAL:
    #     filename = os.path.join("/home/jack/图片", current_time.strftime("%Y%m%d_%H%M%S.jpg"))
    #     cv2.imwrite(filename, frame)
    #     print(f"图片{num}已保存: {filename}")
    #     last_capture_time = current_time 
    #     num += 1
    
    # ESC键退出
    if key == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()