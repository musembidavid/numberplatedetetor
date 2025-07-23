import cv2
import pytesseract


img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 11, 17, 17)  
edges = cv2.Canny(blur, 30, 200)
cntours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntours = sorted(cntours, key=cv2.contourArea, reverse=True)[:10]

plate_contour = None


for i, c in enumerate(cntours):
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4:
        plate_contour = approx
        break




x, y, w, h = cv2.boundingRect(plate_contour)
plate = gray[y:y+h, x:x+w]
text = pytesseract.image_to_string(plate)


print("detected license plate text:", text.strip())



