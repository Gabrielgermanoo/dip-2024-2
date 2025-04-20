import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

imgs = {
    "chips": "../../img/chips.png",
    "lena": "../../img/lena.png",
    "rgb": "../../img/rgb.png",
    "rgbcube_KBKG": "../../img/rgbcube_KBKG.png",
    "flowers": "../../img/flowers.jpg",
    "hsv_disk": "../../img/hsv_disk.png",
    "monkey": "../../img/monkey.jpeg",
    "strawberries": "../../img/strawberries.tif"
}

def display_rgb_histograms(img_path, bins=256):
    
    img = np.array(Image.open(img_path))
    
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(r_channel.flatten(), bins=bins, color='red', alpha=0.7)
    axes[0].set_title('Red Channel Histogram')
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(g_channel.flatten(), bins=bins, color='green', alpha=0.7)
    axes[1].set_title('Green Channel Histogram')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    
    axes[2].hist(b_channel.flatten(), bins=bins, color='blue', alpha=0.7)
    axes[2].set_title('Blue Channel Histogram')
    axes[2].set_xlabel('Pixel Intensity')
    axes[2].set_ylabel('Frequency')
    
    img_name = os.path.basename(img_path)
    fig.suptitle(f'RGB Histograms: {img_name}', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def compare_histograms(natural_img_key, synthetic_img_key, bins=256):
    
    print(f"Comparing natural image ({natural_img_key}) with synthetic image ({synthetic_img_key}):")
    
    
    print("\nNatural image histograms:")
    display_rgb_histograms(imgs[natural_img_key], bins)
    
    # Display histograms for synthetic image
    print("\nSynthetic image histograms:")
    display_rgb_histograms(imgs[synthetic_img_key], bins)
    

# Example

print("Displaying RGB histograms for flowers.png")
display_rgb_histograms(imgs["flowers"])
    

compare_histograms("flowers", "rgb")