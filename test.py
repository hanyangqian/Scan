from PIL import Image
import pytesseract
import cv2
import os

image = cv2.imread("res.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
preprocess = 'thresh' #thresh  #做预处理选项
if preprocess == 'blur':
    gray = cv2.blur(gray,(3,3))
if preprocess == 'thresh':
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)


text = pytesseract.image_to_string(Image.open("res.jpg"))  # 转化成文本
print("======STEP 4 OCR识别======")
print(text)