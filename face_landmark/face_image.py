import cv2
import numpy as np 
import dlib 

from math import hypot


camara = cv2.VideoCapture(0)

naris_cerdo = cv2.imread("nose_pig.png")

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


        naris_arriba = (puntos_referencia.part(29).x, puntos_referencia.part(29).y)
        naris_centro = (puntos_referencia.part(30).x, puntos_referencia.part(30).y)
        naris_izquierda = (puntos_referencia.part(31).x, puntos_referencia.part(31).y)
        naris_derecha = (puntos_referencia.part(35).x, puntos_referencia.part(35).y)
        # cv2.circle(cuadro,naris_arriba,5,(255,255,0),-1)
        # cv2.circle(cuadro,naris_derecha,5,(255,0,0),-1)



        naris_ancho = int(hypot(naris_izquierda[0] - naris_derecha[0],
                                naris_izquierda[1] - naris_derecha[1])*1.8)
        naris_altura = int(naris_ancho * 0.77)

        punto_izquierdo  = (int(naris_centro[0] - naris_ancho / 2),
                                 int(naris_centro[1] - naris_altura / 2 ) )

        punto_derecho = (int(naris_centro[0] + naris_ancho /2),
                                 int(naris_centro[1] + naris_altura /2))                     

        # dibuja un rectangulo al rededor de la naris
        # cv2.rectangle(cuadro, (int(naris_centro[0] - naris_ancho / 2),
        #                         int(naris_centro[1] - naris_altura / 2 ) ),
        #                         (int(naris_centro[0] + naris_ancho /2),
        #                         int(naris_centro[1] + naris_altura /2)),
        #                         (0,225,0), 2)

        # adicionando la nueva naris
        naris_cerdo_redimencionada = cv2.resize(naris_cerdo, (naris_ancho,naris_altura))

        # 

        naris_cerdo_gris = cv2.cvtColor(naris_cerdo_redimencionada, cv2.COLOR_BGR2GRAY)
        _,naris_mascara = cv2.threshold(naris_cerdo_gris, 25, 255, cv2.THRESH_BINARY_INV)
        # 
        area_naris = cuadro[punto_izquierdo[1]: punto_izquierdo[1] + naris_altura,
                            punto_izquierdo[0]: punto_izquierdo[0] + naris_ancho]

        # 

        naris_area_invertida = cv2.bitwise_and(area_naris, area_naris, mask=naris_mascara)


        # 

        naris_final = cv2.add(naris_area_invertida, naris_cerdo_redimencionada)

        cuadro[punto_izquierdo[1]: punto_izquierdo[1] + naris_altura,
                            punto_izquierdo[0]: punto_izquierdo[0] + naris_ancho] = naris_final
        # print(naris_ancho)
        # print(naris_altura)

        

        # se utiliza para marcar todos los puntos de la cara
        # for n in range(0,68):
        #     x = puntos_referencia.part(n).x
        #     y = puntos_referencia.part(n).y 
        #     cv2.circle(cuadro, (x,y), 6, (255, 0, 0), -1)

        # fin ...se utiliza para marcar todos los puntos de la cara
        # print(puntos_referencia)

    # print(caras)
    # cv2.imshow("area naris", area_naris)
    # cv2.imshow("naris mascara", naris_mascara)
    cv2.imshow("Frame", cuadro)
    # cv2.imshow("Cerdo", naris_cerdo_redimencionada)
    # cv2.imshow("invertida", naris_area_invertida)
    # cv2.imshow("final", naris_final)

    key = cv2.waitKey(1)
    if key == 27:
        break
