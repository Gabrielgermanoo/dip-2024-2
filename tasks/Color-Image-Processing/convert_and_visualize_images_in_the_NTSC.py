import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rgb_to_yiq(rgb_image):
    rgb_norm = rgb_image.astype(float) / 255.0
    conversion_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.275, -0.321],
        [0.212, -0.523, 0.311]
    ])
    height, width, _ = rgb_norm.shape
    pixels = rgb_norm.reshape(-1, 3)
    yiq_pixels = np.dot(pixels, conversion_matrix.T)
    yiq_image = yiq_pixels.reshape(height, width, 3)
    return yiq_image

def yiq_to_rgb(yiq_image):
    inverse_matrix = np.array([
        [1.000, 0.956, 0.621],
        [1.000, -0.272, -0.647],
        [1.000, -1.106, 1.703]
    ])
    height, width, _ = yiq_image.shape
    pixels = yiq_image.reshape(-1, 3)
    rgb_pixels = np.dot(pixels, inverse_matrix.T)
    rgb_pixels = np.clip(rgb_pixels, 0, 1)
    rgb_image = rgb_pixels.reshape(height, width, 3)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def visualize_yiq(image_path):
    try:
        img = np.array(Image.open(image_path))
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    yiq = rgb_to_yiq(img)
    y_channel = yiq[:,:,0]
    i_channel = yiq[:,:,1]
    q_channel = yiq[:,:,2]
    reconstructed = yiq_to_rgb(yiq)
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original RGB Image")
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(y_channel, cmap='gray')
    plt.title("Y Channel (Luminance)")
    plt.axis('off')
    y_only = np.zeros_like(yiq)
    y_only[:,:,0] = y_channel
    y_rgb = yiq_to_rgb(y_only)
    plt.subplot(2, 3, 3)
    plt.imshow(y_rgb)
    plt.title("Y Channel Only (as RGB)")
    plt.axis('off')
    plt.subplot(2, 3, 4)
    i_normalized = (i_channel - np.min(i_channel)) / (np.max(i_channel) - np.min(i_channel))
    plt.imshow(i_normalized, cmap='RdBu')
    plt.title("I Channel (R-Y Chrominance)")
    plt.axis('off')
    plt.subplot(2, 3, 5)
    q_normalized = (q_channel - np.min(q_channel)) / (np.max(q_channel) - np.min(q_channel))
    plt.imshow(q_normalized, cmap='PRGn')
    plt.title("Q Channel (B-Y Chrominance)")
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(reconstructed)
    plt.title("Reconstructed from YIQ")
    plt.axis('off')
    plt.tight_layout()
    plt.suptitle("RGB to YIQ Color Space Conversion", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.figure(figsize=(15, 8))
    i_only = np.zeros_like(yiq)
    i_only[:,:,1] = i_channel
    i_rgb = yiq_to_rgb(i_only)
    plt.subplot(2, 3, 1)
    plt.imshow(i_rgb)
    plt.title("I Channel Component")
    plt.axis('off')
    q_only = np.zeros_like(yiq)
    q_only[:,:,2] = q_channel
    q_rgb = yiq_to_rgb(q_only)
    plt.subplot(2, 3, 2)
    plt.imshow(q_rgb)
    plt.title("Q Channel Component")
    plt.axis('off')
    yi = np.zeros_like(yiq)
    yi[:,:,0] = y_channel
    yi[:,:,1] = i_channel
    yi_rgb = yiq_to_rgb(yi)
    plt.subplot(2, 3, 3)
    plt.imshow(yi_rgb)
    plt.title("Y + I Channels")
    plt.axis('off')
    yq = np.zeros_like(yiq)
    yq[:,:,0] = y_channel
    yq[:,:,2] = q_channel
    yq_rgb = yiq_to_rgb(yq)
    plt.subplot(2, 3, 4)
    plt.imshow(yq_rgb)
    plt.title("Y + Q Channels")
    plt.axis('off')
    iq = np.zeros_like(yiq)
    iq[:,:,1] = i_channel
    iq[:,:,2] = q_channel
    iq_rgb = yiq_to_rgb(iq)
    plt.subplot(2, 3, 5)
    plt.imshow(iq_rgb)
    plt.title("I + Q Channels (No Y)")
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(reconstructed)
    plt.title("Full YIQ (Y+I+Q)")
    plt.axis('off')
    plt.tight_layout()
    plt.suptitle("YIQ Channel Combinations", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()
    print("\nYIQ Channel Statistics:")
    print(f"Y Channel - Min: {y_channel.min():.3f}, Max: {y_channel.max():.3f}, Mean: {y_channel.mean():.3f}")
    print(f"I Channel - Min: {i_channel.min():.3f}, Max: {i_channel.max():.3f}, Mean: {i_channel.mean():.3f}")
    print(f"Q Channel - Min: {q_channel.min():.3f}, Max: {q_channel.max():.3f}, Mean: {q_channel.mean():.3f}")
    mse = np.mean((img.astype(float) - reconstructed.astype(float)) ** 2)
    print(f"\nReconstruction MSE: {mse:.2f}")


image_path = "../../img/flowers.jpg"
visualize_yiq(image_path)
