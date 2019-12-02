import cv2
import numpy as np 
import dlib 


camara = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, cuadro = camara.read()
    gray = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)

    caras = detector(gray)

    for cara in caras:
        x1 = cara.left()
        y1 = cara.top()
        x2 = cara.right()
        y2 = cara.bottom()

        # cv2.circle(cuadro, (x2,y2), 2, (0,255,255), -1)
        # cv2.circle(cuadro, (x1,y1), 2, (0,255,0), -1)
        # cv2.rectangle(cuadro, (x1,y1), (x2,y2), (0, 255, 0), 3)

        puntos_referencia = predictor(gray, cara)

        for n in range(0,68):
            x = puntos_referencia.part(n).x
            y = puntos_referencia.part(n).y 
            cv2.circle(cuadro, (x,y), 6, (255, 0, 0), -1)
        # print(puntos_referencia)

    # print(caras)

    cv2.imshow("Frame", cuadro)

    key = cv2.waitKey(1)
    if key == 27:
        break
