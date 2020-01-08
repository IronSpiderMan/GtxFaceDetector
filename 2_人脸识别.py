import cv2

# 加载训练数据集
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')

# 准备识别的图片
im = cv2.imread('1.jpg')
grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# 检测人脸
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = face_detector.detectMultiScale(grey)

for x, y, w, h in face:
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

    label, confidence = recognizer.predict(grey[y:y+h, x:x+w])

    # print(
    #     """
    #     识别到人脸信息
    #     标签：%d
    #     可信度：%d
    #     """%(label, confidence)
    # )

    if confidence <= 60:
        if label % 10 == 1:
            print("小罗伯特唐尼")
            # print("图片标签：", str(label))
            # print("可信度：", str(confidence))
    else:
        print("未匹配到数据")
    cv2.imshow('im', im)
    cv2.waitKey(0)

cv2.destroyAllWindows()