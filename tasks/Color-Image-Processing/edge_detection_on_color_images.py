import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_sobel_filter(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    height, width = img.shape
    gradient_x = np.zeros_like(img, dtype=float)
    gradient_y = np.zeros_like(img, dtype=float)
    padded = np.pad(img, 1, mode='reflect')

    for y in range(height):
        for x in range(width):
            gradient_x[y, x] = np.sum(padded[y:y+3, x:x+3] * sobel_x)
            gradient_y[y, x] = np.sum(padded[y:y+3, x:x+3] * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    if gradient_magnitude.max() > 0:
        gradient_magnitude /= gradient_magnitude.max()

    return gradient_magnitude

def apply_laplacian_filter(img):
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    height, width = img.shape
    result = np.zeros_like(img, dtype=float)
    padded = np.pad(img, 1, mode='reflect')

    for y in range(height):
        for x in range(width):
            result[y, x] = np.sum(padded[y:y+3, x:x+3] * laplacian)

    result = np.abs(result)
    if result.max() > 0:
        result /= result.max()

    return result

def apply_color_edge_detection(image_path):
    img = np.array(Image.open(image_path))
    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    gray_sobel = apply_sobel_filter(gray)
    gray_laplacian = apply_laplacian_filter(gray)

    r_channel, g_channel, b_channel = img[:,:,0], img[:,:,1], img[:,:,2]
    r_sobel, g_sobel, b_sobel = apply_sobel_filter(r_channel), apply_sobel_filter(g_channel), apply_sobel_filter(b_channel)
    r_laplacian, g_laplacian, b_laplacian = apply_laplacian_filter(r_channel), apply_laplacian_filter(g_channel), apply_laplacian_filter(b_channel)

    combined_sobel = np.maximum.reduce([r_sobel, g_sobel, b_sobel])
    combined_laplacian = np.maximum.reduce([r_laplacian, g_laplacian, b_laplacian])

    weighted_sobel = r_sobel * 0.3 + g_sobel * 0.59 + b_sobel * 0.11
    weighted_laplacian = r_laplacian * 0.3 + g_laplacian * 0.59 + b_laplacian * 0.11

    # Original and grayscale images
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(gray, cmap='gray')
    axs[0, 1].set_title("Grayscale")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(gray_sobel, cmap='gray')
    axs[1, 0].set_title("Grayscale Sobel")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(gray_laplacian, cmap='gray')
    axs[1, 1].set_title("Grayscale Laplacian")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # Red channel
    fig, axs = plt.subplots(1, 3, figsize=(8, 5))
    axs[0].imshow(r_channel, cmap='Reds')
    axs[0].set_title("Red Channel")
    axs[0].axis('off')

    axs[1].imshow(r_sobel, cmap='gray')
    axs[1].set_title("Red Sobel", color='red')
    axs[1].axis('off')

    axs[2].imshow(r_laplacian, cmap='gray')
    axs[2].set_title("Red Laplacian", color='red')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Green channel
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(g_channel, cmap='Greens')
    axs[0].set_title("Green Channel")
    axs[0].axis('off')

    axs[1].imshow(g_sobel, cmap='gray')
    axs[1].set_title("Green Sobel", color='green')
    axs[1].axis('off')

    axs[2].imshow(g_laplacian, cmap='gray')
    axs[2].set_title("Green Laplacian", color='green')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Blue channel
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(b_channel, cmap='Blues')
    axs[0].set_title("Blue Channel")
    axs[0].axis('off')

    axs[1].imshow(b_sobel, cmap='gray')
    axs[1].set_title("Blue Sobel", color='blue')
    axs[1].axis('off')

    axs[2].imshow(b_laplacian, cmap='gray')
    axs[2].set_title("Blue Laplacian", color='blue')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Combined results
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(combined_sobel, cmap='gray')
    plt.title("Combined Sobel (Maximum)", color='red')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(combined_laplacian, cmap='gray')
    plt.title("Combined Laplacian (Maximum)", color='red')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(weighted_sobel, cmap='gray')
    plt.title("Weighted Sobel", color='red')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(weighted_laplacian, cmap='gray')
    plt.title("Weighted Laplacian", color='red')
    plt.axis('off')

    color_edge_map = np.zeros_like(img)
    color_edge_map[:,:,0] = r_sobel * 255
    color_edge_map[:,:,1] = g_sobel * 255
    color_edge_map[:,:,2] = b_sobel * 255

    plt.subplot(2, 3, 5)
    plt.imshow(color_edge_map)
    plt.title("Combined Color Edge Map")
    plt.axis('off')

    overlay = img.copy().astype(float)
    mask = combined_sobel > 0.2
    for c in range(3):
        overlay[:,:,c] = overlay[:,:,c] * (1 - mask) + 255 * mask

    plt.subplot(2, 3, 6)
    plt.imshow(overlay.astype(np.uint8))
    plt.title("Edge Overlay on Original")
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Combined Edge Detection Results", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

img_path = "../../img/flowers.jpg"
apply_color_edge_detection(img_path)
