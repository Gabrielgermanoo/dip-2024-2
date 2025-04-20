import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb

def rgb_to_hsv(rgb_img):
    rgb_normalized = rgb_img.astype(float) / 255.0
    r, g, b = rgb_normalized[:, :, 0], rgb_normalized[:, :, 1], rgb_normalized[:, :, 2]
    v = np.max(rgb_normalized, axis=2)
    min_rgb = np.min(rgb_normalized, axis=2)
    delta = v - min_rgb

    s = np.zeros_like(v)
    non_zero = v > 0
    s[non_zero] = delta[non_zero] / v[non_zero]

    h = np.zeros_like(v)
    red_max = (v == r) & (delta != 0)
    h[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6
    green_max = (v == g) & (delta != 0)
    h[green_max] = 2.0 + (b[green_max] - r[green_max]) / delta[green_max]
    blue_max = (v == b) & (delta != 0)
    h[blue_max] = 4.0 + (r[blue_max] - g[blue_max]) / delta[blue_max]

    h = h / 6.0
    hsv_img = np.stack([h, s, v], axis=2)
    return hsv_img

def hsv_to_rgb_custom(hsv_img):
    return hsv_to_rgb(hsv_img)

def create_gaussian_kernel(size=5, sigma=1.0):
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def apply_gaussian_blur(img, kernel_size=5, sigma=1.0):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    result = np.zeros_like(img, dtype=float)

    for i in range(img.shape[2]):
        padded = np.pad(img[:, :, i], kernel_size // 2, mode='reflect')
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                result[y, x, i] = np.sum(
                    padded[y:y+kernel_size, x:x+kernel_size] * kernel
                )
    return result

def compare_blur_effects(image_path, kernel_size=9, sigma=2.0):
    img = np.array(Image.open(image_path))
    img_float = img.astype(float) / 255.0

    blurred_rgb = apply_gaussian_blur(img_float, kernel_size, sigma)
    blurred_rgb = np.clip(blurred_rgb, 0, 1)

    hsv_img = rgb_to_hsv(img)
    blurred_hsv = apply_gaussian_blur(hsv_img, kernel_size, sigma)

    blurred_hsv[:, :, 0] = np.clip(blurred_hsv[:, :, 0], 0, 1)
    blurred_hsv[:, :, 1] = np.clip(blurred_hsv[:, :, 1], 0, 1)
    blurred_hsv[:, :, 2] = np.clip(blurred_hsv[:, :, 2], 0, 1)

    blurred_hsv_to_rgb = hsv_to_rgb_custom(blurred_hsv)

    diff_image = np.abs(blurred_rgb - blurred_hsv_to_rgb) * 5
    diff_image = np.clip(diff_image, 0, 1)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img / 255.0)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(blurred_rgb)
    plt.title("Gaussian Blur in RGB Space")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(blurred_hsv_to_rgb)
    plt.title("Gaussian Blur in HSV Space")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(diff_image)
    plt.title("Difference (5x amplified)")
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle(
        f"Blur Comparison (kernel={kernel_size}, sigma={sigma})", fontsize=16
    )
    plt.subplots_adjust(top=0.9)
    plt.show()

img_path = "../../img/lena.png"
compare_blur_effects(img_path, kernel_size=11, sigma=3.0)