# Kütüphanelerin eklenmesi
import cv2
import numpy as np



# Haar Cascade kütüphanesinin kullanımı
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# third party cascade
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('nose.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# Resi okuma
img = cv2.imread('kemalsunal1.jpg')


# Yüzün Seçilipp dikdörtgen içine alınması
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
for (x, y, w, h) in faces:
    roi_color = img[y:y + h, x:x + h]
    roi_gray = gray[y:y + h, x:x + h]
    # print(x, y, w, h)
    color = (255, 0, 0)
    stroke = 3
    end_cord_x = x + w
    end_cord_y = y + h
    cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)
    # Gözlerin dikdörtgen içerisine alınması
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), stroke)
        # roi = region of
        roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]

    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (255, 255, 255), stroke)
        roi_nose = gray[ny: ny + nh, nx: nx + nw]

    # Gülüş belirlenmesi
    smiles = smile_cascade.detectMultiScale(roi_color)
    for (ex, ey, ew, eh) in smiles:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), stroke)


# Ekrana Bastırma
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
cv2.imshow('Face Makeup', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

