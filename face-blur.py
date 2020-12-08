import cv2
import numpy as np
import face_recognition

# Init
img = cv2.imread('Photos/group2.jpg')

ratio = img.shape[0] / img.shape[1]
width = 900
height = int(width * ratio)
img = cv2.resize(img, (width, height))

mask = np.zeros(img.shape[:2], dtype='uint8')

# Detect faces
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(rgb_img)

# Create Mask
for (top, right, bottom, left) in face_locations:
    # cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    center = ((right+left)//2, (top+bottom)//2)
    radius = (bottom-top)//2 + 10
    cv2.circle(mask, center, radius, 255, -1)

# Blur
blurred_img = cv2.blur(img, (19, 19))
result = img.copy()
result[mask>0] = blurred_img[mask>0]

# Display
cv2.imshow('Face blur', result)
cv2.waitKey(0)
