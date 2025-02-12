import cv2 as cv

# 1.获取视频对象
cap = cv.VideoCapture('demo/input/phone_video.mp4')

# 2.判断是否读取成功
while(cap.isOpened()):

    # 3.获取每一帧图像
    ret, frame = cap.read()
    print(cap.get(3),cap.get(4))

    # 4. 获取成功显示图像
    if ret == True:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)    # BGR 转 RGB
        cv.imshow('frame', rgb_frame)
        cv.waitKey(25)

    # 5.每一帧间隔为25ms
    else:
        break

# 6.释放视频对象
cap.release()
cv.destroyAllWindows()
