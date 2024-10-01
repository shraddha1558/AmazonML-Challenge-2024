import os
import pandas as pd
import cv2
from PIL import Image
import requests
from io import BytesIO
import pytesseract

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
import numpy as np

def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()

    # Check if the image is already in grayscale (1 channel)
    if len(newImage.shape) == 3:
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    else:
        gray = newImage
    
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) > 0:
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)
        
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle
    
    return 0.0  # If no contours found, return 0

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)

def preprocess_image(image):
    # Convert to grayscale
    # gray = grayscale(image)
    
    # Remove noise
    gray = noise_removal(image)
    
    # Thicken font
    gray = thick_font(gray)
    
    # Deskew image
    gray = deskew(gray)
    
    # Remove borders
    # gray = remove_borders(gray)
    
    return gray

def download_image(image_link):
    response = requests.get(image_link)
    image = Image.open(BytesIO(response.content))
    image = np.array(image)
    cv2.imwrite("downloaded.jpg", image)
    return image

def predictor(image_link, category_id, entity_name):
    image = download_image(image_link)
    preprocessed_image = preprocess_image(image)
    ocr_result = pytesseract.image_to_string(preprocessed_image)
    return ocr_result

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)