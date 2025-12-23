import cv2

def histogram_equalization(image):
    """
    Apply Histogram Equalization on grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized
