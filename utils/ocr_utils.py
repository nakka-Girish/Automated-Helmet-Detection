# utils/ocr_utils.py
import cv2
import re
from easyocr import Reader

# Initialize the OCR reader once
reader = Reader(['en'])

def extract_license_plate(image):
    """
    Run OCR on the given image and extract license plate text.
    :param image: Cropped license plate image.
    :return: Cleaned license plate string.
    """
    # Convert image to grayscale (if necessary)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, mag_ratio=3)
    plate_text = ""
    for res in results:
        # Each result is a tuple: (bbox, text, confidence)
        plate_text += res[1]
    # Clean the text
    plate_text = plate_text.replace(" ", "").upper()
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)
    return plate_text
