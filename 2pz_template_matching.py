import cv2
import numpy as np

# Load the image and convert it to grayscale
image = cv2.imread('image/8.jpg')
#image = cv2.resize(image, (827, 1063))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the template and get its dimensions
template = cv2.imread('template/1.jpg', 0)
w, h = template.shape[::-1]

# Perform template matching
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.6
loc = np.where(res >= threshold)

# Draw rectangles around the detected faces
for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# Show the result
image = cv2.resize(image, (551, 708))
cv2.imshow('Template', template)
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()