import os

path = "./gtx/"
paths = os.listdir(path)
i = 0
for file in paths:
    i += 1
    try:
        os.rename(path + file, path + str(i) + str(1) + ".jpg")
    except Exception as e:
        print(e)
        continue