import cv2
import os
import numpy

root_path = "./gtx/"

def getFacesAndLabels():
    """读取图片特征和标签"""
    global root_path
    faces = []
    lables = []

    # 获取人脸检测器
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 获取图片路径
    files = os.listdir(root_path)
    for file in files:
        path = root_path + file

        # 读取图片
        im = cv2.imread(path)
        # 转换灰度
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # 读取人脸数据
        face = face_detector.detectMultiScale(grey)
        for x, y, w, h in face:
            # 设置标签
            lables.append(int(file.split('.')[0]))

            # 设置人脸数据
            faces.append(grey[y:y+h, x:x+w])

    return faces, lables

faces, labels = getFacesAndLabels()

# 获取训练对象
recognizer = cv2.face.LBPHFaceRecognizer_create()


# 训练数据
recognizer.train(faces, numpy.array(labels))
# 保存训练数据
recognizer.write('./trainer/trainer.yml')
