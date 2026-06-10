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
        tuple: (compressed_channels, original_shape)
            - compressed_channels (dict): Dictionary containing compressed data for each channel
            - original_shape (tuple): Original image shape (height, width, channels)
    """
    # Dictionary to store compressed data for each channel
    compressed_channels = {}
    original_shape = None
    
    # For each channel, retain only the specified number of components
    for channel in pca_channel:
        pca, fit_pca = pca_channel[channel]
        # Retain only the first n_components rows (pixels)
        pca_pixel = fit_pca[:, :n_components]
        # Retain only the first n_components principal components
        pca_comp = pca.components_[:n_components, :]
        # Store compressed data along with the mean for later reconstruction
        compressed_channels[channel] = (pca_pixel, pca_comp, pca.mean_)
        # Capture original shape from the first channel
        if original_shape is None:
            original_shape = (fit_pca.shape[0], pca_comp.shape[1], len(pca_channel))
        
    return compressed_channels, original_shape

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


def build_original_npz_data(img_path):
    """
    Build a dictionary of data to save as original.npz.
    
    Args:
        img_path (str): Path to the original image file
        
    Returns:
        dict: Data dictionary with pixel_array and shape metadata
    """
    img = Image.open(img_path).convert('RGB')
    pixel_array = np.array(img)
    return {
        "pixel_array": pixel_array,
        "original_shape": pixel_array.shape
    }


def build_compressed_npz_data(compressed_channels, n_components, original_shape):
    """
    Build a dictionary of data to save as compressed.npz.
    
    Args:
        compressed_channels (dict): Compressed PCA state per channel
        n_components (int): Number of PCA components retained
        original_shape (tuple): Original image shape (height, width, channels)
        
    Returns:
        dict: Data dictionary with PCA state and shape metadata
    """
    data = {
        "n_components": n_components,
        "original_shape": original_shape
    }
    for ch in compressed_channels:
        pca_pixel, pca_comp, pca_mean = compressed_channels[ch]
        data[f"channel_{ch}_pixel"] = pca_pixel
        data[f"channel_{ch}_comp"] = pca_comp
        data[f"channel_{ch}_mean"] = pca_mean
    return data


def rebuild_compressed_channels(npz_data):
    """
    Rebuild the compressed_channels dict from .npz data.
    
    Args:
        npz_data: Loaded .npz file object
        
    Returns:
        tuple: (compressed_channels, n_components, original_shape)
    """
    compressed_channels = {}
    n_components = int(npz_data["n_components"])
    original_shape = tuple(npz_data["original_shape"])
    
    for ch in range(original_shape[2]):
        pca_pixel = npz_data[f"channel_{ch}_pixel"]
        pca_comp = npz_data[f"channel_{ch}_comp"]
        pca_mean = npz_data[f"channel_{ch}_mean"]
        compressed_channels[ch] = (pca_pixel, pca_comp, pca_mean)
    
    return compressed_channels, n_components, original_shape


def compare_images(img_path, compressed_img_array, compressed_channels, n_components, original_shape):
    """
    Display side-by-side comparison of original and compressed images with size information
    measured from .npz file sizes (not JPEG).
    
    Args:
        img_path (str): Path to the original image file
        compressed_img_array (numpy.ndarray): Array containing compressed image pixel data
        compressed_channels (dict): Compressed PCA state per channel
        n_components (int): Number of PCA components used for compression
        original_shape (tuple): Original image shape (height, width, channels)
    """
    # Get original image
    img_open = Image.open(img_path)
    if img_open.mode != 'RGB':
        img_open = img_open.convert('RGB')
    original_pixels = np.array(img_open)
    
    # Convert compressed array to PIL Image for display
    compressed_img = Image.fromarray(compressed_img_array)
    
    # --- Measure sizes using .npz files (not JPEG) ---
    original_temp = "/tmp/temp_original.npz"
    compressed_temp = "/tmp/temp_compressed.npz"
    
    # Build and save original as .npz
    original_data = {"pixel_array": original_pixels, "original_shape": original_pixels.shape}
    np.savez_compressed(original_temp, **original_data)
    
    # Build and save compressed as .npz
    compressed_data = build_compressed_npz_data(compressed_channels, n_components, original_shape)
    np.savez_compressed(compressed_temp, **compressed_data)
    
    # Get file sizes for comparison
    original_size_kb = os.stat(original_temp).st_size / 1024
    compressed_size_kb = os.stat(compressed_temp).st_size / 1024
    
    # Cleanup temporary files
    os.remove(original_temp)
    os.remove(compressed_temp)
    
    # Calculate compression metrics
    size_reduction_kb = original_size_kb - compressed_size_kb
    reduction_percent = (1 - compressed_size_kb / original_size_kb) * 100 if original_size_kb > 0 else 0
    compression_ratio = original_size_kb / compressed_size_kb if compressed_size_kb > 0 else 0
    
    # Create side-by-side comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display original image with size information (from .npz)
    axes[0].imshow(img_open)
    axes[0].set_title(f"Original Image\n.npz Size: {original_size_kb:.2f} KB", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display compressed image with size information and component count
    axes[1].imshow(compressed_img)
    axes[1].set_title(f"Compressed Image ({n_components} components)\n.npz Size: {compressed_size_kb:.2f} KB", 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add summary text with compression metrics
    fig.text(0.5, 0.02, 
             f"Size Reduction: {size_reduction_kb:.2f} KB ({reduction_percent:.1f}%) | Compression Ratio: {compression_ratio:.2f}x",
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout and display the comparison
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def save_to_export(original_npz_data, compressed_npz_data):
    """
    Save original and compressed .npz data to the /export/ directory.
    
    Args:
        original_npz_data (dict): Data for original.npz
        compressed_npz_data (dict): Data for compressed.npz
    """
    export_dir = "./export/"
    os.makedirs(export_dir, exist_ok=True)
    
    orig_path = os.path.join(export_dir, "original.npz")
    comp_path = os.path.join(export_dir, "compressed.npz")
    
    np.savez_compressed(orig_path, **original_npz_data)
    np.savez_compressed(comp_path, **compressed_npz_data)
    
    orig_size = os.stat(orig_path).st_size / 1024
    comp_size = os.stat(comp_path).st_size / 1024
    
    print(f"Saved to /export/:")
    print(f"  {orig_path} ({orig_size:.2f} KB)")
    print(f"  {comp_path} ({comp_size:.2f} KB)")


def decompress_and_view():
    """
    Let the user select a .npz file and display the reconstructed image on screen.
    Works with both original.npz and compressed.npz files.
    """
    filepath = crossfiledialog.open_file()
    if not filepath:
        print("No file selected.")
        return
    
    if not filepath.endswith(".npz"):
        print("Selected file is not a .npz file.")
        return
    
    data = np.load(filepath)
    
    # Detect type: if it has 'pixel_array' key -> original; otherwise compressed
    if "pixel_array" in data:
        # Original .npz — display directly
        pixel_array = data["pixel_array"]
        img = Image.fromarray(pixel_array)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Original Image — {os.path.basename(filepath)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        print(f"Displayed original image from: {filepath}")
    elif "n_components" in data:
        # Compressed .npz — reconstruct via PCA then display
        compressed_channels, n_components, original_shape = rebuild_compressed_channels(data)
        reconstructed = decompress_image_channels_components(compressed_channels)
        img = Image.fromarray(reconstructed)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Decompressed Image ({n_components} components) — {os.path.basename(filepath)}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        print(f"Displayed decompressed image from: {filepath} (shape: {reconstructed.shape})")
    else:
        print(f"Unknown .npz format: {filepath}")
        print("Available keys:", list(data.keys()))


def compress_mode():
    """Run the compression workflow."""
    N_components = 10

    # Ask user to select a photo using file picker
    filepath = crossfiledialog.open_file()
    if not filepath:
        print("No file selected.")
        return
    
    # Process image into separate channels and compute PCA for each
    pca_channel = image_channels_components(filepath)
    
    # Compress image by retaining only N_components principal components per channel
    compressed_pca, original_shape = compress_image_channels_components(pca_channel, N_components)
    
    # Decompress the compressed image data to reconstruct the image
    compressed_img = decompress_image_channels_components(compressed_pca)
    
    # Show comparison between original and compressed images with size reduction metrics
    compare_images(filepath, compressed_img, compressed_pca, N_components, original_shape)
    
    # Ask user if they want to save
    save_choice = input("\nSave original and compressed data to /export/? (y/n): ").strip().lower()
    if save_choice == 'y':
        original_npz_data = build_original_npz_data(filepath)
        compressed_npz_data = build_compressed_npz_data(compressed_pca, N_components, original_shape)
        save_to_export(original_npz_data, compressed_npz_data)


def decompress_mode():
    """Run the decompress-to-view workflow."""
    print("Select a .npz file to decompress and view on screen.")
    decompress_and_view()


def main():
    """
    Main function to execute the image compression workflow.
    """
    print("=== Image Compression with PCA (Pure Math) ===")
    print("Choose mode:")
    print("  1 — Compress an image")
    print("  2 — Decompress a .npz file to view")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        compress_mode()
    elif choice == "2":
        decompress_mode()
    else:
        print(f"Invalid choice: '{choice}'. Please enter 1 or 2.")


# Execute main function when script is run directly
if __name__ == "__main__":
    main()