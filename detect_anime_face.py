import glob
import cv2
import sys
import os.path


def loadImages():
    dataset = []

    for f in glob.glob(r"*.jpg"):  # glob可查找並列出符合路徑的檔案
        image = cv2.imread(f, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        dataset.append((gray, image, f))

    return dataset


def detect(dataset, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    for d in dataset:
        cascade = cv2.CascadeClassifier(cascade_file)

        faces = cascade.detectMultiScale(d[0],
                                         # detector options
                                         scaleFactor=1.01,
                                         minNeighbors=3,
                                         minSize=(24, 24))

        for (x, y, w, h) in faces:
            cropped_img = d[1][y: y + h, x: x + w]
            cv2.imwrite(d[2], cropped_img)


loadImages()
detect(loadImages())
