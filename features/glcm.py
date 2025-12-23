import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    """
    Extract GLCM texture features from an image.
    Image input: BGR (OpenCV format)
    Output: dictionary of GLCM features
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute GLCM
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = {
        "Contrast": graycoprops(glcm, "contrast")[0, 0],
        "Dissimilarity": graycoprops(glcm, "dissimilarity")[0, 0],
        "Homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "Energy": graycoprops(glcm, "energy")[0, 0],
        "Correlation": graycoprops(glcm, "correlation")[0, 0],
    }

    return features

