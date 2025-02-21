import argparse
import numpy as np
import cv2 as cv
import requests


def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    url_response = requests.get(url)
    
    url_response.raise_for_status()
    
    flags = kwargs.get('flags', cv.IMREAD_COLOR)
    image = cv.imdecode(np.frombuffer(url_response.content, np.uint8), flags)
    
    return image

url = "https://picsum.photos/id/237/200/300"

image = load_image_from_url(url)

image = cv.resize(image, (400, 600))

# save image
cv.imwrite("./output/image.jpg", image)