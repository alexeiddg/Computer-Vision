import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import re

# Read the image
image = cv2.imread('./images/plate1.jpg')
original_image = image.copy()

# Check if the image was loaded properly
if image is None:
    print("Error: Image not found or cannot be opened.")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply noise reduction using bilateral filter
bfilter = cv2.bilateralFilter(gray_image, 11, 17, 17)

# Edge detection using Canny edge detector
edged = cv2.Canny(bfilter, 50, 200)

# Find contours in the edged image
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to hold potential character contours
char_contours = []

for idx, contour in enumerate(contours):
    if hierarchy[0][idx][3] != -1:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        if cv2.contourArea(contour) > 100:
            char_contours.append(contour)

# Draw character contours for debugging
char_image = original_image.copy()
cv2.drawContours(char_image, char_contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB))
plt.title('Character Contours')
plt.axis('off')
plt.show()

# Assuming the license plate region has been detected and cropped
# For demonstration, we'll use the entire image
cropped_image = original_image  # Replace with actual cropped image

# Preprocess the cropped image
cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
cropped_gray = cv2.convertScaleAbs(cropped_gray, alpha=2.0, beta=0)
cropped_thresh = cv2.adaptiveThreshold(
    cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 10
)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cropped_thresh = cv2.morphologyEx(cropped_thresh, cv2.MORPH_CLOSE, kernel)
cropped_thresh = cv2.resize(cropped_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Use EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(cropped_thresh, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')

if len(result) == 0:
    print("No text found by EasyOCR.")
else:
    texts = [res[-2] for res in result]
    full_text = ' '.join(texts)
    print("Detected text:", full_text)

    # Define the license plate pattern
    pattern = r'[A-Z]{1,2}-\d{2}-\d{3}'
    match = re.search(pattern, full_text)
    if match:
        license_plate = match.group()
        print("License Plate Number:", license_plate)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Put the detected license plate number on the image
        res = cv2.putText(original_image, text=license_plate, org=(50, 50), fontFace=font,
                          fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # Display the final result
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.title('Final Result')
        plt.axis('off')
        plt.show()
    else:
        print("License plate number not found in OCR results.")
