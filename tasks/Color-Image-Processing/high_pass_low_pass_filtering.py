import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def apply_frequency_domain_filtering(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path))
            # Convert from RGB to BGR for OpenCV compatibility
            img = img[:, :, ::-1]
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Split into channels
    b, g, r = cv2.split(img)
    channels = [b, g, r]
    channel_names = ['Blue', 'Green', 'Red']
    
    
    # Parameters for filters
    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain
    
    # Define radius values for filters (can be adjusted)
    low_pass_radius = min(rows, cols) // 8
    high_pass_radius = min(rows, cols) // 4
    
    # Process each channel
    high_pass_results = []
    low_pass_results = []
    
    for i, (channel, name) in enumerate(zip(channels, channel_names)):
        # Apply padding (optimal for Fourier Transform)
        padded = np.zeros((rows, cols), dtype=np.float32)
        padded[:rows, :cols] = channel
        
        # Perform DFT
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Calculate magnitude spectrum for visualization (log scale)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        # Create high-pass filter mask (Gaussian)
        high_pass_mask = np.ones((rows, cols, 2), np.float32)
        for x in range(rows):
            for y in range(cols):
                distance = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
                high_pass_mask[x, y] = 1 - np.exp(-(distance ** 2) / (2 * high_pass_radius ** 2))
        
        # Create low-pass filter mask (Gaussian)
        low_pass_mask = np.zeros((rows, cols, 2), np.float32)
        for x in range(rows):
            for y in range(cols):
                distance = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
                low_pass_mask[x, y] = np.exp(-(distance ** 2) / (2 * low_pass_radius ** 2))
        
        # Apply high-pass filter
        high_pass_result = dft_shift * high_pass_mask
        high_pass_ishift = np.fft.ifftshift(high_pass_result)
        high_pass_img = cv2.idft(high_pass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        high_pass_img = np.clip(high_pass_img, 0, 255).astype(np.uint8)
        high_pass_results.append(high_pass_img)
        
        # Apply low-pass filter
        low_pass_result = dft_shift * low_pass_mask
        low_pass_ishift = np.fft.ifftshift(low_pass_result)
        low_pass_img = cv2.idft(low_pass_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        low_pass_img = np.clip(low_pass_img, 0, 255).astype(np.uint8)
        low_pass_results.append(low_pass_img)
        
    # Reconstruct and visualize filtered images
    high_pass_img = cv2.merge(high_pass_results)
    low_pass_img = cv2.merge(low_pass_results)
    
    # Display original and filtered images side by side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(high_pass_img, cv2.COLOR_BGR2RGB))
    plt.title('High-Pass Filter (Edge Enhancement)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(low_pass_img, cv2.COLOR_BGR2RGB))
    plt.title('Low-Pass Filter (Smoothing)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
image_path = "../../img/strawberries.tif"
apply_frequency_domain_filtering(image_path)