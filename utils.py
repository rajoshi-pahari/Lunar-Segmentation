import io
from PIL import Image
from skimage.io import imread
import numpy as np
import tensorflow as tf


def process_input_image(image_path):
    """
    Prepares an image for a segmentation model by resizing, cropping, 
    and normalizing it to [0, 1].

    Parameters:
    - image_path: Path or file-like object of the input image.

    Returns:
    - np.ndarray: The processed image array with dimensions (480, 480, 3).
    """
    # Target size for the model
    target_height, target_width = 480, 480

    # Load the image
    image = imread(image_path)

    # Validate dimensions and color channels
    height, width, channels = image.shape
    if height < target_height or width < target_width:
        raise ValueError(f"Image must be at least {target_height}x{target_width}. "
                         f"Provided: {image.shape}")
    if channels != 3:
        raise ValueError(f"Expected 3 color channels (RGB). Found: {channels}")

    # Crop the image to target dimensions
    cropped_image = image[:target_height, :target_width, :]

    # Normalize to the range [0, 1]
    normalized_image = cropped_image / 255.0
    float_image = normalized_image.astype(np.float32)

    return float_image


def map_colors():
    """
    Returns a color mapping for different classes in the segmented output.

    Returns:
    - np.ndarray: Array of RGB values for each class.
    """
    return np.array([
        [0, 0, 0],       # Class 0: Black (Background)
        [255, 0, 0],     # Class 1: Red (Large Rocks)
        [0, 255, 0],     # Class 2: Green (Sky)
        [0, 0, 255]      # Class 3: Blue (Small Rocks)
    ], dtype=np.uint8)