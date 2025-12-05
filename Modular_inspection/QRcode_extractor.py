#extract qrcode from substrate image and return the qr code if it has one 
import cv2
import numpy 

img = cv2.imread("test.png")    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
approx = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)
cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()