import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import math

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

def rgb_to_lab(rgb_img):
    
    rgb = rgb_img.astype(float) / 255.0
    
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    
    M = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    
    xyz = np.dot(rgb.reshape(-1, 3), M.T).reshape(rgb.shape)
    
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / xyz_ref
    
    mask = xyz > 0.008856
    xyz[mask] = np.power(xyz[mask], 1/3)
    xyz[~mask] = 7.787 * xyz[~mask] + 16/116
    
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    lab = np.stack([L, a, b], axis=2)
    return lab

def rgb_to_ycrcb(rgb_img):
    rgb = rgb_img.astype(float) / 255.0
    
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.5 + (r - y) * 0.713
    cb = 0.5 + (b - y) * 0.564
    
    ycrcb = np.stack([y, cr, cb], axis=2)
    return ycrcb

def rgb_to_cmyk(rgb_img):
    rgb = rgb_img.astype(float) / 255.0
    
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    k = 1 - np.maximum(np.maximum(r, g), b)
    
    k_mask = (k == 1)
    
    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y = np.zeros_like(k)
    
    c[~k_mask] = (1 - r[~k_mask] - k[~k_mask]) / (1 - k[~k_mask])
    m[~k_mask] = (1 - g[~k_mask] - k[~k_mask]) / (1 - k[~k_mask])
    y[~k_mask] = (1 - b[~k_mask] - k[~k_mask]) / (1 - k[~k_mask])
    
    cmyk = np.stack([c, m, y, k], axis=2)
    return cmyk

def visualize_color_spaces(image_path):

    img = np.array(Image.open(image_path))

    hsv_img = rgb_to_hsv(img)
    lab_img = rgb_to_lab(img)
    ycrcb_img = rgb_to_ycrcb(img)
    cmyk_img = rgb_to_cmyk(img)
    
    plt.figure(figsize=(20, 15))
    
    plt.subplot(5, 4, 1)
    plt.imshow(img)
    plt.title("Original RGB")
    plt.axis('off')

    plt.subplot(5, 4, 2)
    plt.imshow(img[:,:,0], cmap='Reds')
    plt.title("R Channel")
    plt.axis('off')
    
    plt.subplot(5, 4, 3)
    plt.imshow(img[:,:,1], cmap='Greens')
    plt.title("G Channel")
    plt.axis('off')
    
    plt.subplot(5, 4, 4)
    plt.imshow(img[:,:,2], cmap='Blues')
    plt.title("B Channel")
    plt.axis('off')
    
    plt.subplot(5, 4, 5)
    hsv_rgb = hsv_to_rgb(hsv_img)
    plt.imshow(hsv_rgb)
    plt.title("HSV")
    plt.axis('off')
    
    plt.subplot(5, 4, 6)
    plt.imshow(hsv_img[:,:,0], cmap='hsv')
    plt.title("Hue")
    plt.axis('off')
    
    plt.subplot(5, 4, 7)
    plt.imshow(hsv_img[:,:,1], cmap='gray')
    plt.title("Saturation")
    plt.axis('off')
    
    plt.subplot(5, 4, 8)
    plt.imshow(hsv_img[:,:,2], cmap='gray')
    plt.title("Value")
    plt.axis('off')

    plt.subplot(5, 4, 9)
    l_norm = (lab_img[:,:,0] - lab_img[:,:,0].min()) / (lab_img[:,:,0].max() - lab_img[:,:,0].min())
    a_norm = (lab_img[:,:,1] - lab_img[:,:,1].min()) / (lab_img[:,:,1].max() - lab_img[:,:,1].min())
    b_norm = (lab_img[:,:,2] - lab_img[:,:,2].min()) / (lab_img[:,:,2].max() - lab_img[:,:,2].min())
    lab_vis = np.stack([l_norm, a_norm, b_norm], axis=2)
    plt.imshow(lab_vis)
    plt.title("LAB")
    plt.axis('off')
    
    plt.subplot(5, 4, 10)
    plt.imshow(l_norm, cmap='gray')
    plt.title("L (Lightness)")
    plt.axis('off')
    
    plt.subplot(5, 4, 11)
    plt.imshow(a_norm, cmap='RdYlGn_r')
    plt.title("a (Green-Red)")
    plt.axis('off')
    
    plt.subplot(5, 4, 12)
    plt.imshow(b_norm, cmap='coolwarm')
    plt.title("b (Blue-Yellow)")
    plt.axis('off')

    plt.subplot(5, 4, 13)
    ycrcb_vis = np.clip(ycrcb_img, 0, 1)
    plt.imshow(ycrcb_vis)
    plt.title("YCrCb")
    plt.axis('off')
    
    plt.subplot(5, 4, 14)
    plt.imshow(ycrcb_img[:,:,0], cmap='gray')
    plt.title("Y (Luminance)")
    plt.axis('off')
    
    plt.subplot(5, 4, 15)
    plt.imshow(ycrcb_img[:,:,1], cmap='Reds')
    plt.title("Cr (Red-diff)")
    plt.axis('off')
    
    plt.subplot(5, 4, 16)
    plt.imshow(ycrcb_img[:,:,2], cmap='Blues')
    plt.title("Cb (Blue-diff)")
    plt.axis('off')
    
    plt.subplot(5, 4, 17)
    cmyk_vis = np.stack([1-cmyk_img[:,:,0], 1-cmyk_img[:,:,1], 1-cmyk_img[:,:,2]], axis=2)
    plt.imshow(cmyk_vis)
    plt.title("CMYK")
    plt.axis('off')
    
    plt.subplot(5, 4, 18)
    plt.imshow(cmyk_img[:,:,0], cmap='Blues')
    plt.title("Cyan")
    plt.axis('off')
    
    plt.subplot(5, 4, 19)
    plt.imshow(cmyk_img[:,:,1], cmap='magma')
    plt.title("Magenta")
    plt.axis('off')
    
    plt.subplot(5, 4, 20)
    plt.imshow(cmyk_img[:,:,2], cmap='YlOrBr')
    plt.title("Yellow")
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Color Space Conversions: {image_path.split('/')[-1]}", fontsize=16)
    plt.subplots_adjust(top=0.94)
    plt.show()
    
img_path = "../../img/chips.png"
visualize_color_spaces(img_path)