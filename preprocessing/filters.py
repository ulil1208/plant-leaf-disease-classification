import cv2
import numpy as np

def gaussian_filter(image, kernel_size=(5, 5)):
    """
    Apply Gaussian Blur to reduce noise
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def median_filter(image, kernel_size=5):
    """
    Apply Median Blur to remove salt-and-pepper noise
    """
    return cv2.medianBlur(image, kernel_size)

