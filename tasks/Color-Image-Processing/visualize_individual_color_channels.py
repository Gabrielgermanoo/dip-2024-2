import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_color_channels(image_path):
    
    img = np.array(Image.open(image_path))
    
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]
    
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.imshow(r_channel, cmap='gray')
    ax2.set_title('Red Channel (Grayscale)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(g_channel, cmap='gray')
    ax3.set_title('Green Channel (Grayscale)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.imshow(b_channel, cmap='gray')
    ax4.set_title('Blue Channel (Grayscale)')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(3, 4, 5)
    red_img = np.zeros_like(img)
    red_img[:, :, 0] = r_channel
    ax5.imshow(red_img)
    ax5.set_title('Red Channel (Colored)')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(3, 4, 6)
    green_img = np.zeros_like(img)
    green_img[:, :, 1] = g_channel
    ax6.imshow(green_img)
    ax6.set_title('Green Channel (Colored)')
    ax6.axis('off')

    ax7 = fig.add_subplot(3, 4, 7)
    blue_img = np.zeros_like(img)
    blue_img[:, :, 2] = b_channel
    ax7.imshow(blue_img)
    ax7.set_title('Blue Channel (Colored)')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(3, 4, 8)
    rg_img = np.zeros_like(img)
    rg_img[:, :, 0] = r_channel
    rg_img[:, :, 1] = g_channel
    ax8.imshow(rg_img)
    ax8.set_title('Red + Green')
    ax8.axis('off')
    
    ax9 = fig.add_subplot(3, 4, 9)
    rb_img = np.zeros_like(img)
    rb_img[:, :, 0] = r_channel
    rb_img[:, :, 2] = b_channel
    ax9.imshow(rb_img)
    ax9.set_title('Red + Blue')
    ax9.axis('off')
    
    ax10 = fig.add_subplot(3, 4, 10)
    gb_img = np.zeros_like(img)
    gb_img[:, :, 1] = g_channel
    gb_img[:, :, 2] = b_channel
    ax10.imshow(gb_img)
    ax10.set_title('Green + Blue')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(3, 4, 11)
    reconstructed = np.zeros_like(img)
    reconstructed[:, :, 0] = r_channel
    reconstructed[:, :, 1] = g_channel
    reconstructed[:, :, 2] = b_channel
    ax11.imshow(reconstructed)
    ax11.set_title('Reconstructed Image')
    ax11.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Color Channel Visualization: {image_path.split('/')[-1]}", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    is_identical = np.array_equal(img, reconstructed)
    print(f"Reconstruction successful: {is_identical}")
    
    if not is_identical:
        diff = np.abs(img.astype(int) - reconstructed.astype(int)).sum()
        print(f"Total pixel difference: {diff}")

image_path = "../../img/monkey.jpeg"
visualize_color_channels(image_path)