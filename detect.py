import cv2
from matplotlib import pyplot as plt
fig = plt.figure()

plt.rcParams['figure.figsize'] = (224, 224)

face = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("videos/video.mp4")

i = 0
j = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        HistFull = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.hist(HistFull)
        j += 1
        plt.savefig("Hist/frame{}.png".format(j))

        faces = face.detectMultiScale(
            gray,
            scaleFactor=[1.1],
            minNeighbors=[5],
            minSize=[20, 20],
        )

        for (x, y, w, h) in faces:
            aa = frame[y:y + h, x:x + w]
            i += 1
            cv2.imwrite("Pessoas/face{}.jpg".format(i), aa)
            gray1 = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY)
            HistFull2 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            plt.hist(HistFull2)
            plt.savefig("HistFace/frame{}.png".format(i))
    else:
        break

cap.release()
cv2.destroyAllWindows()