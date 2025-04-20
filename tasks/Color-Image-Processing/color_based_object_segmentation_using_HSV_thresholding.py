import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

def segment_by_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image from {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([0, 100, 70])
    upper_bound = np.array([10, 255, 255])
    
    mask1 = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    lower_bound2 = np.array([160, 100, 70])
    upper_bound2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_bound2, upper_bound2)
    
    mask = mask1 + mask2
    
    segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(segmented_img)
    plt.title("Segmented Objects")
    plt.axis('off')
    
    h, s, v = cv2.split(img_hsv)
    
    plt.subplot(2, 3, 4)
    plt.imshow(h, cmap='hsv')
    plt.title("Hue Channel")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(s, cmap='gray')
    plt.title("Saturation Channel")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(v, cmap='gray')
    plt.title("Value Channel")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def interactive_segmentation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image from {image_path}")
        return
    
    cv2.namedWindow('Original')
    cv2.namedWindow('Segmented')
    cv2.namedWindow('Controls')
    
    cv2.createTrackbar('Hue Min', 'Controls', 0, 179, nothing)
    cv2.createTrackbar('Hue Max', 'Controls', 10, 179, nothing)
    cv2.createTrackbar('Sat Min', 'Controls', 100, 255, nothing)
    cv2.createTrackbar('Sat Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('Val Min', 'Controls', 70, 255, nothing)
    cv2.createTrackbar('Val Max', 'Controls', 255, 255, nothing)
    
    cv2.createTrackbar('Color Presets', 'Controls', 0, 5, nothing)
    
    cv2.imshow('Original', img)
    
    while True:
        h_min = cv2.getTrackbarPos('Hue Min', 'Controls')
        h_max = cv2.getTrackbarPos('Hue Max', 'Controls')
        s_min = cv2.getTrackbarPos('Sat Min', 'Controls')
        s_max = cv2.getTrackbarPos('Sat Max', 'Controls')
        v_min = cv2.getTrackbarPos('Val Min', 'Controls')
        v_max = cv2.getTrackbarPos('Val Max', 'Controls')
        
        preset = cv2.getTrackbarPos('Color Presets', 'Controls')
        
        if preset == 1:
            cv2.setTrackbarPos('Hue Min', 'Controls', 0)
            cv2.setTrackbarPos('Hue Max', 'Controls', 10)
            cv2.setTrackbarPos('Sat Min', 'Controls', 100)
            cv2.setTrackbarPos('Sat Max', 'Controls', 255)
            cv2.setTrackbarPos('Val Min', 'Controls', 70)
            cv2.setTrackbarPos('Val Max', 'Controls', 255)
            cv2.setTrackbarPos('Color Presets', 'Controls', 0)
        elif preset == 2:
            cv2.setTrackbarPos('Hue Min', 'Controls', 35)
            cv2.setTrackbarPos('Hue Max', 'Controls', 85)
            cv2.setTrackbarPos('Sat Min', 'Controls', 50)
            cv2.setTrackbarPos('Sat Max', 'Controls', 255)
            cv2.setTrackbarPos('Val Min', 'Controls', 50)
            cv2.setTrackbarPos('Val Max', 'Controls', 255)
            cv2.setTrackbarPos('Color Presets', 'Controls', 0)
        elif preset == 3:
            cv2.setTrackbarPos('Hue Min', 'Controls', 90)
            cv2.setTrackbarPos('Hue Max', 'Controls', 130)
            cv2.setTrackbarPos('Sat Min', 'Controls', 100)
            cv2.setTrackbarPos('Sat Max', 'Controls', 255)
            cv2.setTrackbarPos('Val Min', 'Controls', 70)
            cv2.setTrackbarPos('Val Max', 'Controls', 255)
            cv2.setTrackbarPos('Color Presets', 'Controls', 0)
        elif preset == 4:
            cv2.setTrackbarPos('Hue Min', 'Controls', 20)
            cv2.setTrackbarPos('Hue Max', 'Controls', 35)
            cv2.setTrackbarPos('Sat Min', 'Controls', 100)
            cv2.setTrackbarPos('Sat Max', 'Controls', 255)
            cv2.setTrackbarPos('Val Min', 'Controls', 100)
            cv2.setTrackbarPos('Val Max', 'Controls', 255)
            cv2.setTrackbarPos('Color Presets', 'Controls', 0)
        elif preset == 5:
            cv2.setTrackbarPos('Hue Min', 'Controls', 10)
            cv2.setTrackbarPos('Hue Max', 'Controls', 25)
            cv2.setTrackbarPos('Sat Min', 'Controls', 150)
            cv2.setTrackbarPos('Sat Max', 'Controls', 255)
            cv2.setTrackbarPos('Val Min', 'Controls', 150)
            cv2.setTrackbarPos('Val Max', 'Controls', 255)
            cv2.setTrackbarPos('Color Presets', 'Controls', 0)
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        if h_min < 10 and h_max < 20:
            lower2 = np.array([160, s_min, v_min])
            upper2 = np.array([179, s_max, v_max])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        
        result = cv2.bitwise_and(img, img, mask=mask)
        
        cv2.imshow('Mask', mask)
        cv2.imshow('Segmented', result)
        
        h, s, v = cv2.split(hsv)
        
        h_disp = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
        s_disp = s
        v_disp = v
        
        hsv_display = np.hstack([h_disp, s_disp, v_disp])
        cv2.imshow('HSV Channels (H, S, V)', hsv_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "../../img/red_apples.jpg"
    interactive_segmentation(image_path)
