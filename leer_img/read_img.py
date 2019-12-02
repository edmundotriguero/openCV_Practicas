import cv2

imagen = cv2.imread('python.jpeg',0)
cv2.imshow('prueba de gloria!!!', imagen)
cv2.imwrite('grises.jpg', imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()