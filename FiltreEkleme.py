import cv2
import imutils

# Haar Cascade kütüphanesinin kullanımı
from cv2.cv2 import CascadeClassifier

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# third party cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# Resi okuma
img = cv2.imread('kemalsunal1.jpg')
ruj = cv2.imread('lip.png')
image_model = cv2.imread('sapka.png')
kirpik = cv2.imread('eyelash.png')

sakal = cv2.imread('eyebrow.png', -1)
bıyık = cv2.imread('mustache.png', -1)
gözlük = cv2.imread('glasses.png', -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

shiftValueW = 25
shiftValueH = 35
shiftValueX = 0
shiftValueY = -10
bgColorThresholdValue = 230

for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + h]
    roi_color = img[y:y + h, x:x + h]
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
    x = x + shiftValueX
    y = y + shiftValueY
    model_width = int(0.85 * w) + shiftValueW
    model_height = int(0.50 * h) + 20
    sapka = cv2.resize(image_model, (model_width, model_height))
    for i in range(0, model_height):
        for j in range(0, model_width):
            for k in range(3):
                if sapka[i][j][k] < bgColorThresholdValue:
                    img[y + i - int(0.25 * h)][x + j][k] = sapka[i][j][k]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
        roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
        glasses2 = imutils.resize(gözlük, width=ew + 40, height=eh + 40)

        gw, gh, gc = glasses2.shape
        for i in range(0, gw):
            for j in range(0, gh):
                # print(glasses[i, j]) #RGBA
                if glasses2[i, j][3] != 0:  # alpha 0
                    roi_color[ey + i, ex + j] = glasses2[i, j]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        kirpik = cv2.resize(kirpik, (ew, eh))
        for i in range(0, eh):
            for j in range(0, ew):
                for k in range(3):
                    if kirpik[i][j][k] < bgColorThresholdValue:
                        img[y + i - int(0.25 * h) + 75][x + j + 72][k] = kirpik[i][j][k]
                        img[y + i - int(0.25 * h) + 75][x + j + 30][k] = kirpik[i][j][k]

    nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    for (nx, ny, nw, nh) in nose:
        # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
        roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
        mustache2 = imutils.resize(bıyık.copy(), width=nw)

        mw, mh, mc = mustache2.shape
        for i in range(0, mw):
            for j in range(0, mh):
                if mustache2[i, j][3] != 0:  # alpha 0
                    roi_color[ny + int(nh / 2.0) + i, nx + j] = mustache2[i, j]

    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    for (sx, sy, sw, sh) in smiles:
    # cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 3)
        ruj = cv2.resize(ruj, (sw-32, sh-18))
        for i in range(0, sh-18):
            for j in range(0, sw-32):
                for k in range(3):
                    if ruj[i][j][k] < bgColorThresholdValue:
                        img[sw + i - int(0.25 * h) + 52][x + j + 2 + 34][k] = ruj[i][j][k]

img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
cv2.imshow('Face Makeup', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
