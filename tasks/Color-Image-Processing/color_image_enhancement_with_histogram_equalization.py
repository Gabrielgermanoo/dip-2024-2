import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_histogram_equalization(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            img_pil = Image.open(image_path)
            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(16, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    img_rgb_eq = np.zeros_like(img_rgb)
    for i in range(3):
        img_rgb_eq[:,:,i] = cv2.equalizeHist(img_rgb[:,:,i])
    
    plt.subplot(3, 3, 2)
    plt.imshow(img_rgb_eq)
    plt.title("RGB Direct Equalization\n(Problematic)")
    plt.axis('off')
    
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y_eq = cv2.equalizeHist(y)
    img_ycrcb_eq = cv2.merge([y_eq, cr, cb])
    img_ycrcb_eq_rgb = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    
    plt.subplot(3, 3, 3)
    plt.imshow(img_ycrcb_eq_rgb)
    plt.title("YCrCb Equalization\n(Y Channel Only)")
    plt.axis('off')
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l_eq = cv2.equalizeHist(l)
    img_lab_eq = cv2.merge([l_eq, a, b])
    img_lab_eq_rgb = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
    
    plt.subplot(3, 3, 4)
    plt.imshow(img_lab_eq_rgb)
    plt.title("LAB Equalization\n(L Channel Only)")
    plt.axis('off')
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v_eq = cv2.equalizeHist(v)
    img_hsv_eq = cv2.merge([h, s, v_eq])
    img_hsv_eq_rgb = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2RGB)
    
    plt.subplot(3, 3, 5)
    plt.imshow(img_hsv_eq_rgb)
    plt.title("HSV Equalization\n(V Channel Only)")
    plt.axis('off')
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe_rgb = np.zeros_like(img_rgb)
    for i in range(3):
        img_clahe_rgb[:,:,i] = clahe.apply(img_rgb[:,:,i])
    
    plt.subplot(3, 3, 6)
    plt.imshow(img_clahe_rgb)
    plt.title("CLAHE on RGB\n(Not Recommended)")
    plt.axis('off')
    
    y_clahe = clahe.apply(y)
    img_ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    img_ycrcb_clahe_rgb = cv2.cvtColor(img_ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    
    plt.subplot(3, 3, 7)
    plt.imshow(img_ycrcb_clahe_rgb)
    plt.title("CLAHE on YCrCb\n(Y Channel)")
    plt.axis('off')
    
    l_clahe = clahe.apply(l)
    img_lab_clahe = cv2.merge([l_clahe, a, b])
    img_lab_clahe_rgb = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
    
    plt.subplot(3, 3, 8)
    plt.imshow(img_lab_clahe_rgb)
    plt.title("CLAHE on LAB\n(L Channel)")
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.hist(y.flatten(), 256, [0, 256], color='gray', alpha=0.5, label='Original Y')
    plt.hist(y_eq.flatten(), 256, [0, 256], color='blue', alpha=0.5, label='Equalized Y')
    plt.title("Histogram Comparison\n(Y Channel)")
    plt.legend()
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.suptitle("Color Image Enhancement with Histogram Equalization", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    fig, axs = plt.subplots(3, 3, figsize=(16, 12))
    
    for i, color_name in enumerate(['Red', 'Green', 'Blue']):
        axs[0, i].hist(img_rgb[:,:,i].flatten(), 256, [0, 256], color=color_name.lower(), alpha=0.7)
        axs[0, i].set_title(f"Original {color_name}")
    
    axs[1, 0].hist(y.flatten(), 256, [0, 256], color='gray', alpha=0.5, label='Original')
    axs[1, 0].hist(y_eq.flatten(), 256, [0, 256], color='blue', alpha=0.5, label='Equalized')
    axs[1, 0].set_title("Y Channel")
    axs[1, 0].legend()
    
    axs[1, 1].hist(cr.flatten(), 256, [0, 256], color='red')
    axs[1, 1].set_title("Cr Channel (Unchanged)")
    
    axs[1, 2].hist(cb.flatten(), 256, [0, 256], color='blue')
    axs[1, 2].set_title("Cb Channel (Unchanged)")
    
    axs[2, 0].hist(l.flatten(), 256, [0, 256], color='gray', alpha=0.5, label='Original')
    axs[2, 0].hist(l_eq.flatten(), 256, [0, 256], color='blue', alpha=0.5, label='Equalized')
    axs[2, 0].set_title("L Channel")
    axs[2, 0].legend()
    
    axs[2, 1].hist(a.flatten(), 256, [0, 256], color='green')
    axs[2, 1].set_title("a Channel (Unchanged)")
    
    axs[2, 2].hist(b.flatten(), 256, [0, 256], color='orange')
    axs[2, 2].set_title("b Channel (Unchanged)")
    
    plt.tight_layout()
    plt.suptitle("Histogram Analysis Before and After Equalization", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

image_path = "../../img/flowers.jpg"
apply_histogram_equalization(image_path)
