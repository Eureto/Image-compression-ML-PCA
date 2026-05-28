# Import necessary libraries:
# - PIL for image processing
# - os for file system operations
# - numpy for numerical computations
# - matplotlib for plotting
# - crossfiledialog for file selection dialog
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
import crossfiledialog



class PCA:
    """
    Principal Component Analysis implementation using pure mathematical operations.
    
    This class implements PCA from scratch using only numpy for mathematical operations,
    without relying on sklearn or other machine learning libraries.
    """
    
    def __init__(self, random_state=None):
        """
        Initialize PCA object.
        
        Args:
            random_state (int, optional): Random state for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize attributes that will be set during fitting
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        """
        Fit PCA model to the data.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            self: Returns the instance itself
        """
        # Store the mean of the data for centering
        self.mean_ = np.mean(X, axis=0)
        
        # Center the data by subtracting the mean
        X_centered = X - self.mean_
        
        # Compute the covariance matrix
        # cov(X) = (1/(n-1)) * X_centered.T @ X_centered
        n_samples = X.shape[0]
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors of the covariance matrix
        # For symmetric matrices, eigh is more efficient and stable than eig
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        # (eigh returns them in ascending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store the principal components (eigenvectors)
        self.components_ = eigenvectors.T  # Transpose to match sklearn format
        
        # Store explained variance (eigenvalues)
        self.explained_variance_ = eigenvalues
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Transformed data of shape (n_samples, n_components)
        """
        # Center the data using the mean computed during fitting
        X_centered = X - self.mean_
        
        # Project the data onto the principal components
        # This is done by multiplying the centered data with the components matrix
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Transformed data of shape (n_samples, n_components)
        """
        # Fit the model and then transform the data
        self.fit(X)
        return self.transform(X)



# Define constant for sample images folder
_IMAGES_FOLDER = "./sample_images/"

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

def image_channels_components(imgPath):
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


def compress_image_channels_components(pca_channel, n_components):
    """
    Compress image channels by retaining only the specified number of PCA components.
    
    Args:
        pca_channel (dict): Dictionary containing PCA models and transformed data for each channel
        n_components (int): Number of principal components to retain for compression
        
    Returns:
        dict: Dictionary containing compressed data for each channel
    """
    # Dictionary to store compressed data for each channel
    compressed_channels = {}
    
    # For each channel, retain only the specified number of components
    for channel in pca_channel:
        pca, fit_pca = pca_channel[channel]
        # Retain only the first n_components rows (pixels)
        pca_pixel = fit_pca[:, :n_components]
        # Retain only the first n_components principal components
        pca_comp = pca.components_[:n_components, :]
        # Store compressed data along with the mean for later reconstruction
        compressed_channels[channel] = (pca_pixel, pca_comp, pca.mean_)
        
    return compressed_channels

def decompress_image_channels_components(compressed_channels):
    """Reconstruct an image from the compressed PCA representation.

    The original implementation used ``np.transpose`` on a list of channel
    arrays, which swapped axes and produced a mis‑ordered image for non‑square
    inputs. This version stacks the channel arrays along the last axis to
    preserve the original ``(height, width, channels)`` layout.
    """
    channel_arrays = []
    for channel in compressed_channels:
        pca_pixel, pca_comp, pca_mean = compressed_channels[channel]
        compressed_pixels = np.dot(pca_pixel, pca_comp) + pca_mean
        channel_arrays.append(compressed_pixels)

    # Stack channels to shape (height, width, channels)
    decompressed_image = np.stack(channel_arrays, axis=2)
    decompressed_image = np.clip(np.round(decompressed_image), 0, 255).astype(np.uint8)
    return decompressed_image


def compare_images(img_path, compressed_img_array, n_components):
    """
    Display side-by-side comparison of original and compressed images with size information.
    
    Args:
        img_path (str): Path to the original image file
        compressed_img_array (numpy.ndarray): Array containing compressed image pixel data
        n_components (int): Number of PCA components used for compression
    """
    # Get original image
    img_open = Image.open(img_path)
    
    # Convert compressed array to PIL Image for easier handling
    compressed_img = Image.fromarray(compressed_img_array)
    
    # Save both images as JPG with consistent quality for fair file size comparison
    original_temp = "/tmp/temp_original.jpg"
    compressed_temp = "/tmp/temp_compressed.jpg"
    
    # Convert images to RGB mode if needed (JPG doesn't support transparency)
    if img_open.mode != 'RGB':
        img_open = img_open.convert('RGB')
    if compressed_img.mode != 'RGB':
        compressed_img = compressed_img.convert('RGB')
    
    # Save images with consistent quality setting
    img_open.save(original_temp, "JPEG", quality=85)
    compressed_img.save(compressed_temp, "JPEG", quality=85)
    
    # Get file sizes for comparison
    original_size_kb = os.stat(original_temp).st_size / 1024
    compressed_size_kb = os.stat(compressed_temp).st_size / 1024
    
    # Cleanup temporary files
    os.remove(original_temp)
    os.remove(compressed_temp)
    
    # Calculate compression metrics
    size_reduction_kb = original_size_kb - compressed_size_kb
    reduction_percent = (1 - compressed_size_kb / original_size_kb) * 100
    compression_ratio = original_size_kb / compressed_size_kb if compressed_size_kb > 0 else 0
    
    # Create side-by-side comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display original image with size information
    axes[0].imshow(img_open)
    axes[0].set_title(f"Original Image\nSize: {original_size_kb:.2f} KB", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display compressed image with size information and component count
    axes[1].imshow(compressed_img)
    axes[1].set_title(f"Compressed Image ({n_components} components)\nSize: {compressed_size_kb:.2f} KB", 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add summary text with compression metrics
    fig.text(0.5, 0.02, 
             f"Size Reduction: {size_reduction_kb:.2f} KB ({reduction_percent:.1f}%) | Compression Ratio: {compression_ratio:.2f}x",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout and display the comparison
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()



def main():
    """
    Main function to execute the image compression workflow.
    """
    # Ask user to select a photo using file picker
    filepath = crossfiledialog.open_file()
    
    # Process image into separate channels and compute PCA for each
    pca_channel = image_channels_components(filepath)
    
    # Compress image by retaining only 30 principal components per channel
    compressed_pca = compress_image_channels_components(pca_channel, 30)
    
    # Decompress the compressed image data to reconstruct the image
    compressed_img = decompress_image_channels_components(compressed_pca)
    
    # Show comparison between original and compressed images with size reduction metrics
    compare_images(filepath, compressed_img, 30)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()

