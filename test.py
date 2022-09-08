import cv2
from infer import CarDetection


sample_image = cv2.imread("sample_image/vid_4_1840.jpg")
model = CarDetection()

result = model.detect(sample_image)
print(result)