import cv2

def custom_preprocessor(img):
    # img is numpy array (0–255)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img