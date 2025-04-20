import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def extract_bit_planes(img_channel):
    bit_planes = []
    for bit in range(8):  # 8 bits (0-7)
        # Extract bit plane by bitwise AND with a mask
        # where only the target bit is set to 1
        bit_mask = 1 << bit
        bit_plane = (img_channel & bit_mask) > 0
        # Convert to uint8 for visualization (0 or 255)
        bit_plane = bit_plane.astype(np.uint8) * 255
        bit_planes.append(bit_plane)
    
    return bit_planes

def reconstruct_from_bits(bit_planes, bits_to_use):
    reconstructed = np.zeros_like(bit_planes[0], dtype=np.uint8)
    
    for bit in bits_to_use:
        if bit < 0 or bit > 7:
            continue
        # Set the corresponding bit in the reconstructed image
        reconstructed |= ((bit_planes[bit] > 0).astype(np.uint8)) << bit
    
    return reconstructed

def visualize_bit_planes(image_path):

    img = np.array(Image.open(image_path))

    
    # Check if image is RGB
    if len(img.shape) < 3:
        print("Error: Input must be a color image")
        return
    
    # Split into channels
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    
    # Extract bit planes for each channel
    r_bit_planes = extract_bit_planes(r_channel)
    g_bit_planes = extract_bit_planes(g_channel)
    b_bit_planes = extract_bit_planes(b_channel)
    
    # Prepare figure for visualization
    plt.figure(figsize=(16, 12))
    
    # Display original image
    plt.subplot(4, 9, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    
    # Display each channel
    plt.subplot(4, 9, 2)
    plt.imshow(r_channel, cmap='Reds')
    plt.title("Red Channel")
    plt.axis('off')
    
    plt.subplot(4, 9, 3)
    plt.imshow(g_channel, cmap='Greens')
    plt.title("Green Channel")
    plt.axis('off')
    
    plt.subplot(4, 9, 4)
    plt.imshow(b_channel, cmap='Blues')
    plt.title("Blue Channel")
    plt.axis('off')
    
    # Display bit planes for Red channel
    for bit in range(8):
        plt.subplot(4, 9, 10 + bit)
        plt.imshow(r_bit_planes[bit], cmap='gray')
        if bit == 7:
            plt.title(f"Red Bit {bit} (MSB)", color='red')
        elif bit == 0:
            plt.title(f"Red Bit {bit} (LSB)", color='red')
        else:
            plt.title(f"Red Bit {bit}")
        plt.axis('off')
    
    for bit in range(8):
        plt.subplot(4, 9, 19 + bit)
        plt.imshow(g_bit_planes[bit], cmap='gray')
        if bit == 7:
            plt.title(f"Green Bit {bit} (MSB)", color='green')
        elif bit == 0:
            plt.title(f"Green Bit {bit} (LSB)", color='green')
        else:
            plt.title(f"Green Bit {bit}")
        plt.axis('off')
    
    # Display bit planes for Blue channel
    for bit in range(8):
        plt.subplot(4, 9, 28 + bit)
        plt.imshow(b_bit_planes[bit], cmap='gray')
        if bit == 7:
            plt.title(f"Blue Bit {bit} (MSB)", color='blue')
        elif bit == 0:
            plt.title(f"Blue Bit {bit} (LSB)", color='blue')
        else:
            plt.title(f"Blue Bit {bit}")
        plt.axis('off')
    
    # Challenge: Reconstruct with only top 4 bits
    top_bits = [4, 5, 6, 7]  # Top 4 bits
    
    r_reconstructed = reconstruct_from_bits(r_bit_planes, top_bits)
    g_reconstructed = reconstruct_from_bits(g_bit_planes, top_bits)
    b_reconstructed = reconstruct_from_bits(b_bit_planes, top_bits)
    
    # Combine reconstructed channels
    reconstructed_img = np.zeros_like(img)
    reconstructed_img[:,:,0] = r_reconstructed
    reconstructed_img[:,:,1] = g_reconstructed
    reconstructed_img[:,:,2] = b_reconstructed
    
    
    diff = np.abs(img.astype(int) - reconstructed_img.astype(int)).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_img)
    plt.title("Reconstructed (Top 4 Bits Only)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff * 5)  # Enhance the difference
    plt.title("Difference (x5)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print information loss
    mse = np.mean((img.astype(float) - reconstructed_img.astype(float)) ** 2)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Information retained: {(1 - mse/np.mean(img.astype(float)**2)) * 100:.2f}%")
    print(f"Data reduction: From 8 bits to 4 bits per channel = 50% reduction")

image_path = "../../img/hsv_disk.png"
visualize_bit_planes(image_path)
