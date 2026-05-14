from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



_IMAGES_FOLDER = "./sample_images/"

def image_data(imgPath):
    #open image and get the info
    img_size_kb = os.stat(imgPath).st_size/1024
    img_open = Image.open(imgPath)
    data = img_open.getdata()

    #Attempts to reshape the flat pixel data into a 3D array with dimensions (width, height, channels). 
    # If the image is grayscale, the -1 will resolve to 1 (one channel for intensity).
    # If the image is RGB, the -1 will resolve to 3 (three channels for Red, Green, Blue).
    img_pixels = np.array(data).reshape(*img_open.size, -1) 
    img_dim = img_pixels.shape

    dict = {}
    dict["image_size_kb"] = img_size_kb
    dict["image_dimension"] = img_dim

    return dict

def pca_image(imgPath):
    img_open = Image.open(imgPath)
    data = np.array(img_open.getdata())
    img_pixels = data.reshape(*img_open.size, -1)

    pca_channel={}
    t_img = np.transpose(img_pixels)

    for i in range(img_pixels.shape[-1]): #Dla każdego kanału RGB oblicznamy osobne pca
        per_channel = t_img[i]
        channel = t_img[i].reshape(*img_pixels.shape[:-1])
        pca = PCA(random_state = 42)
        fit_pca = pca.fit_transform(channel)
        pca_channel[i] = (pca, fit_pca)

    return pca_channel

def pca_compressed_image(pca_channel, n_components):
    temp_res = []
    for channel in pca_channel:
        pca, fit_pca = pca_channel[channel]
        pca_pixel = fit_pca[:, :n_components]
        pca_comp = pca.components_[:n_components, :]
        compressed_pixels = np.dot(pca_pixel, pca_comp) + pca.mean_
        temp_res.append(compressed_pixels)

    compressed_image = np.transpose(temp_res)
    compressed_image = np.array(compressed_image, dtype=np.uint8)
    return compressed_image


def compare_images(img_path, compressed_img_array, n_components):
    """Display side-by-side comparison of original and compressed images with size information"""
    # Get original image
    img_open = Image.open(img_path)
    
    # Convert compressed array to PIL Image
    compressed_img = Image.fromarray(compressed_img_array)
    
    # Save both as JPG with consistent quality for fair comparison
    original_temp = "/tmp/temp_original.jpg"
    compressed_temp = "/tmp/temp_compressed.jpg"
    
    # Convert grayscale to RGB if needed for JPG
    if img_open.mode != 'RGB':
        img_open = img_open.convert('RGB')
    if compressed_img.mode != 'RGB':
        compressed_img = compressed_img.convert('RGB')
    
    img_open.save(original_temp, "JPEG", quality=85)
    compressed_img.save(compressed_temp, "JPEG", quality=85)
    
    # Get file sizes
    original_size_kb = os.stat(original_temp).st_size / 1024
    compressed_size_kb = os.stat(compressed_temp).st_size / 1024
    
    # Cleanup temp files
    os.remove(original_temp)
    os.remove(compressed_temp)
    
    # Calculate compression metrics
    size_reduction_kb = original_size_kb - compressed_size_kb
    reduction_percent = (1 - compressed_size_kb / original_size_kb) * 100
    compression_ratio = original_size_kb / compressed_size_kb if compressed_size_kb > 0 else 0
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(img_open)
    axes[0].set_title(f"Original Image\nSize: {original_size_kb:.2f} KB", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Compressed image
    axes[1].imshow(compressed_img)
    axes[1].set_title(f"Compressed Image ({n_components} components)\nSize: {compressed_size_kb:.2f} KB", 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add summary text
    fig.text(0.5, 0.02, 
             f"Size Reduction: {size_reduction_kb:.2f} KB ({reduction_percent:.1f}%) | Compression Ratio: {compression_ratio:.2f}x",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()




def main():
    #Get names of images in folder sample_images
    photos_list = os.listdir(_IMAGES_FOLDER)

    photos_list_path = []
    for photo in photos_list:
        photos_list_path.append(_IMAGES_FOLDER + photo)

    for photo in photos_list_path:
        # Process image
        pca_channel = pca_image(photo)
        compressed_img = pca_compressed_image(pca_channel, 30)
        
        # Show comparison with size reduction
        compare_images(photo, compressed_img, 30)





if __name__ == "__main__":
    main()
