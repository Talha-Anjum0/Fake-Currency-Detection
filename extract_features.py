import cv2
import numpy as np

def extract_features(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image to a fixed size
    image = cv2.resize(image, (200, 100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram of pixel intensities
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalize the histogram (makes it model-friendly)
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist
