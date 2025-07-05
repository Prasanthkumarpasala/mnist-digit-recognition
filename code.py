import numpy as np
import struct
import os

def load_mnist_images(file_path):
    """
    Loads and preprocesses MNIST image data from IDX format.

    Args:
        file_path (str): Path to the IDX image file.

    Returns:
        images (np.ndarray): Normalized image data (num_images, 28, 28)
        images_flat (np.ndarray): Flattened image data (num_images, 784)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        # Read metadata
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number: expected 2051, got {magic}")
        
        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Reshape to (num_images, 28, 28)
        images = image_data.reshape((num_images, rows, cols))
        
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Flatten images to (num_images, 784) if needed
        images_flat = images.reshape(num_images, -1)
        
        return images, images_flat

# === USAGE EXAMPLE ===
file_path = "train-images.idx3-ubyte"  # Change path as needed

# Load and preprocess
images, images_flat = load_mnist_images(file_path)

# Confirm shapes
print("Images shape:", images.shape)        # (60000, 28, 28)
print("Flattened shape:", images_flat.shape)  # (60000, 784)

# === OPTIONAL: Save to file ===
np.save("mnist_images.npy", images)
np.save("mnist_images_flat.npy", images_flat)

print("Preprocessed data saved as 'mnist_images.npy' and 'mnist_images_flat.npy'")
