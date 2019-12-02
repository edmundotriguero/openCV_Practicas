import cv2
import numpy as np 
import imutils

image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_,th = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)


img, contornos,herarquia = cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image,contornos, -1, (0,255,0), 3)


cv2.imshow('image',th)
cv2.imshow('image 1',image)
cv2.imshow('image 2',img)





cv2.waitKey(0)
cv2.destroyAllWindows()
