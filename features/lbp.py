import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(image, radius=1, points=8):
    """
    Extract LBP features from an image.
    Image input: BGR
    Output: normalized LBP histogram
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    # Histogram of LBP
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist

