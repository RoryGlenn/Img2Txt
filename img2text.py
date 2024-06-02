from typing import Any
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance

# Set the path to the tesseract executable
PATH_TO_TESSERACT = "C:/Program Files/Tesseract-OCR/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT


def preprocess_image(image_path: str) -> Image.Image:
    """
    Preprocess the image to enhance its quality for OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Image.Image: The preprocessed image.
    """
    img = Image.open(image_path)  # Open the image
    img = img.convert("L")  # Convert to grayscale
    img = img.resize(
        (img.width * 2, img.height * 2), Image.Resampling.LANCZOS
    )  # Resize the image
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Enhance contrast
    img = np.array(img)  # Convert to OpenCV format
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )  # Apply binarization
    img = Image.fromarray(img)  # Convert back to PIL Image
    return img


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from the given image using Tesseract OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text from the image.
    """
    img = preprocess_image(image_path)  # Preprocess the image
    text = pytesseract.image_to_string(img)  # Use pytesseract to extract text
    return text


# Specify the path to your image
image_path = "image.png"

# Extract the text
extracted_text = extract_text_from_image(image_path)
print(extracted_text)

# Save the extracted text to a file
output_path = "extracted_text.txt"
with open(output_path, "w") as file:
    file.write(extracted_text)
