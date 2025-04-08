# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    translated = np.zeros_like(img)
    translated[shift_y:, shift_x:] = img[:img.shape[0] - shift_y, :img.shape[1] - shift_x]
    return translated

def rotate_image_90_clockwise(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, -1)

def stretch_image_horizontally(img: np.ndarray, scale: float) -> np.ndarray:
    stretched = np.zeros((img.shape[0], int(img.shape[1] * scale)))
    for i in range(img.shape[0]):
        stretched[i] = np.interp(
            np.linspace(0, img.shape[1] - 1, stretched.shape[1]),
            np.arange(img.shape[1]),
            img[i]
        )
    return stretched

def mirror_image_horizontally(img: np.ndarray) -> np.ndarray:
    return np.fliplr(img)

def barrel_distort_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    center_x, center_y = w // 2, h // 2
    distorted = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            factor = 1 + 0.0005 * r**2
            src_x = int(center_x + dx / factor)
            src_y = int(center_y + dy / factor)
            if 0 <= src_x < w and 0 <= src_y < h:
                distorted[y, x] = img[src_y, src_x]
    return distorted

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        "translated": translate_image(img, shift_x=10, shift_y=10),
        "rotated": rotate_image_90_clockwise(img),
        "stretched": stretch_image_horizontally(img, scale=1.5),
        "mirrored": mirror_image_horizontally(img),
        "distorted": barrel_distort_image(img)
    }