from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



_IMAGES_FOLDER = "./sample_images/"

# ==================== CONFIGURATION ====================
# Set MANUAL_COMPONENTS to None to use automatic optimal detection
# Set MANUAL_COMPONENTS to an integer (e.g., 50, 100, 150) to override and use fixed components
MANUAL_COMPONENTS = None  # Change this to manually set components (e.g., 50, 100, 150)
# ========================================================

def image_data(imgPath):
    """Return basic information about an image.

    The original implementation assumed the image data could be reshaped
    directly from ``Image.getdata()`` using the image's ``size`` tuple. This
    works for simple RGB JPEGs but fails for other formats (e.g. WebP) and for
    non‑square images because the width/height ordering is swapped later in the
    pipeline. The updated version forces an RGB conversion and uses ``np.array``
    which returns an array with shape ``(height, width, channels)`` – the natural
    layout for Pillow images. This eliminates the need for manual reshaping and
    ensures the dimensions are reported correctly.
    """
    img_size_kb = os.stat(imgPath).st_size / 1024
    # Force RGB to guarantee three channels and a consistent colour space
    img = Image.open(imgPath).convert('RGB')
    img_pixels = np.array(img)  # shape: (height, width, 3)
    img_dim = img_pixels.shape

    return {"image_size_kb": img_size_kb, "image_dimension": img_dim}

def pca_image(imgPath):
    """Perform PCA on each colour channel of an image.

    The function now:
    1. Opens the image and converts it to RGB to guarantee a 3‑channel layout.
    2. Uses ``np.array`` to obtain a ``(height, width, 3)`` array.
    3. Transposes the array to ``(channels, height, width)`` so each channel can
       be processed independently.
    4. Fits a PCA model to each 2‑D channel (treated as ``height`` samples and
       ``width`` features).
    """
    img = Image.open(imgPath).convert('RGB')
    img_pixels = np.array(img)  # (height, width, 3)

    # Move channel axis to the front: (3, height, width)
    t_img = np.transpose(img_pixels, (2, 0, 1))

    pca_channel = {}
    for i in range(t_img.shape[0]):  # iterate over channels
        channel = t_img[i]  # shape: (height, width)
        pca = PCA(random_state=42)
        fit_pca = pca.fit_transform(channel)
        pca_channel[i] = (pca, fit_pca)

    return pca_channel

def pca_compressed_image(pca_channel, n_components):
    """Reconstruct an image from the PCA representation.

    The original implementation built a list of 2‑D channel arrays, then used
    ``np.transpose`` on that list. ``np.transpose`` swaps the first two axes,
    which unintentionally flipped width and height for non‑square images. The
    updated version stacks the channel arrays along the last axis using
    ``np.stack`` which preserves the original ``(height, width)`` layout.
    """
    channel_arrays = []
    for channel in pca_channel:
        pca, fit_pca = pca_channel[channel]
        # Keep only the requested number of components
        pca_pixel = fit_pca[:, :n_components]
        pca_comp = pca.components_[:n_components, :]
        compressed_pixels = np.dot(pca_pixel, pca_comp) + pca.mean_
        channel_arrays.append(compressed_pixels)

    # Stack channels to shape (height, width, channels)
    compressed_image = np.stack(channel_arrays, axis=2)
    compressed_image = np.clip(np.round(compressed_image), 0, 255).astype(np.uint8)
    return compressed_image


def get_optimal_components(pca_channel, variance_threshold=0.95):
    """
    Determine optimal number of components based on explained variance ratio.
    
    Args:
        pca_channel (dict): Dictionary containing PCA models for each channel
        variance_threshold (float): Threshold of explained variance to retain (0-1)
        
    Returns:
        int: Optimal number of components
    """
    max_components_needed = 0
    
    for channel in pca_channel:
        pca, _ = pca_channel[channel]
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        components_needed = np.argmax(cumsum_var >= variance_threshold) + 1
        max_components_needed = max(max_components_needed, components_needed)
    
    return max_components_needed


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

def save_compressed_image(original_path: str, compressed_img_array: np.ndarray) -> None:
    """Save the compressed image to the ``compressed_output`` directory.

    The function creates the output directory if it does not exist and writes the
    compressed image as a JPEG file named ``<original_basename>_compressed.jpg``.
    This allows the user to inspect the result without relying on the interactive
    ``matplotlib`` window, which may not be visible in a headless environment.
    """
    # Ensure the output directory exists
    output_dir = os.path.join(os.getcwd(), "compressed_output")
    os.makedirs(output_dir, exist_ok=True)

    # Derive a filename based on the original image name
    base_name = os.path.basename(original_path)
    name_without_ext, _ = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name_without_ext}_compressed.jpg")

    # Convert the NumPy array back to a PIL Image and save as JPEG
    compressed_img = Image.fromarray(compressed_img_array)
    # Ensure the image is in RGB mode for JPEG compatibility
    if compressed_img.mode != "RGB":
        compressed_img = compressed_img.convert("RGB")
    compressed_img.save(output_path, "JPEG", quality=85)
    print(f"Compressed image saved to: {output_path}")




def main():
    #Get names of images in folder sample_images
    photos_list = os.listdir(_IMAGES_FOLDER)

    photos_list_path = []
    for photo in photos_list:
        photos_list_path.append(_IMAGES_FOLDER + photo)

    for photo in photos_list_path:
        # Process image
        pca_channel = pca_image(photo)
        
        # Determine number of components
        if MANUAL_COMPONENTS is not None:
            # Use manually specified components
            n_components = MANUAL_COMPONENTS
            optimal_components = "N/A (manual override)"
            print(f"Image: {photo} | Using manual override: {n_components} components")
        else:
            # Automatically determine optimal components based on explained variance (95%)
            optimal_components = get_optimal_components(pca_channel, variance_threshold=0.95)
            # Use a minimum of 100 components (or the optimal value if higher) but cap at 200
            n_components = max(min(optimal_components, 200), 100)
            print(f"Image: {photo} | Optimal components: {optimal_components} | Using: {n_components}")
        
        compressed_img = pca_compressed_image(pca_channel, n_components)
        
        # Show comparison with size reduction
        compare_images(photo, compressed_img, n_components)
        # Also save the compressed image to a file for inspection
        save_compressed_image(photo, compressed_img)





if __name__ == "__main__":
    main()
