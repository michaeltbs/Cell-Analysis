"""
Image Analysis Workflow for Multi-Channel Microscopy Data

This module provides functions for processing and analyzing multi-channel microscopy images,
including channel combination, thresholding, segmentation, and cell counting.
"""

from aicsimageio import AICSImage
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from typing import Tuple, Dict, List, Optional
import pandas as pd


def load_image_data(file_path: str, t_index: int = 0, z_index: int = 0) -> Tuple[np.ndarray, List[str]]:
    """
    Load and parse multi-channel microscopy image data.
    
    Args:
        file_path: Path to the .czi image file
        t_index: Time index to extract (default: 0)
        z_index: Z-stack index to extract (default: 0)
        
    Returns:
        Tuple of (image_array, channel_names)
    """
    img = AICSImage(file_path)
    channel_names = img.channel_names
    
    print(f"Channel names: {channel_names}")
    
    # Extract TCZYX array
    arr = img.get_image_data("TCZYX")
    
    print("Array dimensions:")
    for i, dim_size in enumerate(arr.shape):
        print(f"  Dimension {i}: {dim_size}")
    
    return arr, channel_names


def extract_channels(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract individual channels from the multi-channel array.
    
    Args:
        arr: Multi-channel array with shape (T, C, Z, Y, X)
        
    Returns:
        Dictionary mapping channel names to arrays
    """
    channels = {
        'glp1r': arr[:, 0, :, :, :],
        'pomc': arr[:, 1, :, :, :],
        'gal': arr[:, 2, :, :, :],
        'dapi': arr[:, 3, :, :, :] if arr.shape[1] > 3 else None
    }
    
    # Remove None values
    return {k: v for k, v in channels.items() if v is not None}


def multiply_and_scale(image_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Multiply pixel values across channels and apply min-max scaling.
    
    Args:
        image_arrays: List of image arrays to multiply together
        
    Returns:
        Scaled result array with values in range [0,1]
    """
    # Stack arrays and multiply along channel axis
    combined_arr = np.stack(image_arrays, axis=0)
    result_multiplied = np.prod(combined_arr, axis=0)
    
    # Min-Max scaling
    min_val = np.min(result_multiplied)
    max_val = np.max(result_multiplied)
    
    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(result_multiplied)
    else:
        return (result_multiplied - min_val) / (max_val - min_val)


def split_horizontally(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image horizontally at the middle Y-coordinate.
    
    Args:
        image_array: Input image array
        
    Returns:
        Tuple of (upper_half, lower_half)
    """
    max_y = image_array.shape[2]  # Y-dimension
    mid_y = max_y // 2
    
    upper_half = image_array[:, :, :mid_y, :]
    lower_half = image_array[:, :, mid_y:, :]
    
    return upper_half, lower_half


def apply_thresholding(image: np.ndarray, threshold_method: str = 'otsu', 
                      custom_threshold: Optional[float] = None,
                      window_size: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Apply thresholding to create a binary mask. This function now handles both global
    and local thresholding methods and is safe for 16-bit float data.

    Args:
        image (np.ndarray): Input 2D image array.
        threshold_method (str): Method for threshold calculation.
        custom_threshold (Optional[float]): A custom threshold value.
        window_size (Optional[int]): Window size for local thresholding.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the binary mask and the 
                                  calculated threshold value.
    """
    image_2d = np.squeeze(image)
    
    if custom_threshold is not None:
        threshold_value = custom_threshold
    elif threshold_method == 'otsu':
        threshold_value = filters.threshold_otsu(image_2d)
    elif threshold_method == 'yen':
        threshold_value = filters.threshold_yen(image_2d)
    elif threshold_method == 'sauvola':
        if window_size is None:
            raise ValueError("window_size must be provided for Sauvola.")
        threshold_value = filters.threshold_sauvola(image_2d, window_size=window_size)
    elif threshold_method == 'phansalkar':
        if window_size is None:
            raise ValueError("window_size must be provided for Phansalkar.")
        threshold_value = threshold_phansalkar(image_2d, window_size=window_size)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    print(f"Threshold value ({threshold_method}): {np.mean(threshold_value)}")
    
    binary_mask = image_2d > threshold_value
    
    print(f"Binary mask: {np.sum(binary_mask)} True values out of {image_2d.size} total pixels")
    
    return binary_mask, np.mean(threshold_value)


from skimage.util import img_as_ubyte, img_as_float
from scipy.ndimage import uniform_filter

def threshold_phansalkar(image, window_size=7, k=0.25, r=0.5, p=2, q=10):
    """
    Apply Phansalkar local thresholding, adapted for 16-bit float images.

    This version uses scipy's uniform_filter to be safe for float arithmetic and 
    to preserve the full dynamic range of 16-bit data.

    Args:
        image (np.ndarray): Input grayscale image.
        window_size (int): The size of the local window. Must be odd.
        k, r, p, q (float): Phansalkar algorithm parameters.

    Returns:
        np.ndarray: The thresholded image, with the same float type as the input.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    # Normalize the image to [0, 1] for the Phansalkar formula
    image_float = img_as_float(image)
    
    # Calculate local mean using a uniform filter
    mean = uniform_filter(image_float, size=window_size)
    
    # Calculate local standard deviation
    mean_sq = uniform_filter(image_float**2, size=window_size)
    variance = mean_sq - mean**2
    variance[variance < 0] = 0  # Ensure non-negative
    std_dev = np.sqrt(variance)

    # Phansalkar's threshold formula
    threshold = mean * (1 + p * np.exp(-q * mean) + k * ((std_dev / r) - 1))

    # The threshold is already in the same [0, 1] range as image_float.
    # We now scale it back to the original image's range.
    min_val, max_val = np.min(image), np.max(image)
    threshold_scaled = threshold * (max_val - min_val) + min_val
    
    return threshold_scaled


def postprocess_mask(binary_mask: np.ndarray, disk_radius: int = 3) -> np.ndarray:
    """
    Post-process binary mask with hole filling and median filtering.
    
    Args:
        binary_mask: Input binary mask
        disk_radius: Radius for morphological operations
        
    Returns:
        Processed binary mask
    """
    # Fill holes
    filled_mask = ndimage.binary_fill_holes(binary_mask)
    
    pixels_before = np.sum(binary_mask)
    pixels_after = np.sum(filled_mask)
    pixels_filled = pixels_after - pixels_before
    
    print(f"Hole filling: {pixels_filled} pixels filled")
    
    # Median filter
    footprint = morphology.disk(disk_radius)
    filtered_mask = filters.median(filled_mask, footprint=footprint)
    
    return filtered_mask


def watershed_segmentation(binary_mask: np.ndarray, min_distance: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply watershed segmentation to separate touching objects.
    
    Args:
        binary_mask: Binary mask of objects
        min_distance: Minimum distance between watershed markers
        
    Returns:
        Tuple of (labeled_objects, boundaries)
    """
    # Distance transformation
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find markers (peaks)
    markers = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    markers_mask = np.zeros(distance.shape, dtype=bool)
    markers_mask[tuple(markers.T)] = True
    labeled_markers, _ = ndimage.label(markers_mask)
    
    # Watershed segmentation
    labels = segmentation.watershed(-distance, labeled_markers, mask=binary_mask)
    
    # Find boundaries
    boundaries = find_boundaries(labels, mode='thick')
    
    print(f"Watershed segmentation: {labels.max()} objects found")
    
    return labels, boundaries


def analyze_objects(labels: np.ndarray, intensity_image: np.ndarray) -> pd.DataFrame:
    """
    Analyze segmented objects and extract measurements.
    
    Args:
        labels: Labeled segmentation mask
        intensity_image: Original intensity image for measurements
        
    Returns:
        DataFrame with object measurements
    """
    # Ensure 2D arrays
    labels_2d = np.squeeze(labels)
    intensity_2d = np.squeeze(intensity_image)
    
    # Extract properties
    properties = ['label', 'area', 'mean_intensity']
    props = measure.regionprops_table(labels_2d, 
                                    intensity_image=intensity_2d, 
                                    properties=properties)
    
    # Create DataFrame
    results_df = pd.DataFrame(props)
    
    # Calculate integrated density
    results_df['integrated_density'] = results_df['area'] * results_df['mean_intensity']
    
    print(f"Analysis complete: {len(results_df)} objects measured")
    
    return results_df


def save_image_with_boundaries(results: Dict, output_dir: str, filename_prefix: str = "analysis"):
    """
    Save various analysis results as .tif images in a specified directory.
    
    Args:
        results: Dictionary containing all results and intermediate data
        output_dir: Directory path where to save the images
        filename_prefix: Prefix for the output filenames
    """
    import os
    from skimage import io
    from skimage.util import img_as_ubyte
    from skimage.color import label2rgb
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from results dictionary
    image_to_process = results.get('image_to_process')
    processed_mask = results.get('processed_mask') 
    labels = results.get('labels')
    boundaries = results.get('boundaries')
    
    if image_to_process is None:
        raise ValueError("image_to_process not found in results")
    if processed_mask is None:
        raise ValueError("processed_mask not found in results")
    if labels is None:
        raise ValueError("labels not found in results")
    if boundaries is None:
        raise ValueError("boundaries not found in results")
    
    # Normalize images to 8-bit for saving
    # Original image (scaled to 0-255)
    original_8bit = img_as_ubyte((image_to_process - image_to_process.min()) / 
                                (image_to_process.max() - image_to_process.min()))
    
    # Binary mask
    mask_8bit = img_as_ubyte(processed_mask)
    
    # Labels as colored image
    labels_colored = label2rgb(labels, bg_label=0)
    labels_colored_8bit = img_as_ubyte(labels_colored)
    
    # Boundaries overlay on original image
    overlay = image_to_process.copy()
    overlay_normalized = (overlay - overlay.min()) / (overlay.max() - overlay.min())
    
    # Create RGB overlay with boundaries in red
    overlay_rgb = np.stack([overlay_normalized, overlay_normalized, overlay_normalized], axis=-1)
    overlay_rgb[boundaries, 0] = 1.0  # Red channel for boundaries
    overlay_rgb[boundaries, 1] = 0.0  # Green channel
    overlay_rgb[boundaries, 2] = 0.0  # Blue channel
    overlay_rgb_8bit = img_as_ubyte(overlay_rgb)
    
    # Save all images
    file_paths = {}
    
    # 1. Original processed image
    original_path = os.path.join(f"{output_dir}", f"{filename_prefix}_01_original.tif")
    io.imsave(original_path, original_8bit)
    file_paths['original'] = original_path
    
    # 2. Binary mask
    mask_path = os.path.join(output_dir, f"{filename_prefix}_02_binary_mask.tif")
    io.imsave(mask_path, mask_8bit)
    file_paths['mask'] = mask_path
    
    # 3. Segmented labels (colored)
    labels_path = os.path.join(output_dir, f"{filename_prefix}_03_segmented_labels.tif")
    io.imsave(labels_path, labels_colored_8bit)
    file_paths['labels'] = labels_path
    
    # 4. Boundaries overlay
    overlay_path = os.path.join(output_dir, f"{filename_prefix}_04_boundaries_overlay.tif")
    io.imsave(overlay_path, overlay_rgb_8bit)
    file_paths['overlay'] = overlay_path
    
    # 5. Raw boundaries mask
    boundaries_path = os.path.join(output_dir, f"{filename_prefix}_05_boundaries_mask.tif")
    boundaries_8bit = img_as_ubyte(boundaries)
    io.imsave(boundaries_path, boundaries_8bit)
    file_paths['boundaries'] = boundaries_path
    
    # CSV measurements removed as per user request
    
    # Create summary file with analysis parameters
    summary_path = os.path.join(output_dir, f"{filename_prefix}_00_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Image Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Identification:\n")
        f.write(f"- Filename: {results.get('filename', 'unknown')}\n")
        f.write(f"- Channels Used: {', '.join(results.get('use_channels', []))}\n")
        f.write(f"- Region Analyzed: {results.get('region', 'unknown').capitalize()}\n\n")
        
        f.write("Key Metrics:\n")
        if 'measurements' in results:
            measurements = results['measurements']
            f.write(f"- Total objects detected: {len(measurements)}\n")
            f.write(f"- Mean area: {measurements['area'].mean():.2f} pixels\n")
            f.write(f"- Mean intensity: {measurements['mean_intensity'].mean():.2f}\n")
            f.write(f"- Mean integrated density: {measurements['integrated_density'].mean():.2f}\n\n")
        
        f.write("Processing Parameters:\n")
        f.write(f"- Threshold method: {results.get('threshold_method', 'unknown')}\n")
        f.write(f"- Threshold value: {results.get('threshold_value', 'unknown')}\n")
        f.write(f"- Object count: {labels.max() if labels is not None else 'unknown'}\n\n")
        
        f.write("Output Files:\n")
        for key, path in file_paths.items():
            f.write(f"- {key}: {os.path.basename(path)}\n")
    
    file_paths['summary'] = summary_path
    
    print(f"Analysis results saved to: {output_dir}")
    print(f"Files created: {len(file_paths)}")
    for key, path in file_paths.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    return file_paths

def run_workflow(file_path: str, 
                use_channels: List[str] = ['glp1r','pomc', 'gal','dapi'],
                 region: str = 'lower',
                 threshold_method: str = 'otsu',
                 window_size: int = 25,
                 disk_radius: int = 2,
                 min_distance: int = 7) -> Dict:
    """
    Run the complete image analysis workflow.
    
    Args:
        file_path: Path to input image file
        use_channels: List of channel names to combine
        region: Which region to process ('upper', 'lower', 'both')
        threshold_method: Thresholding method to use
        window_size: Window size for local thresholding methods
        disk_radius: Radius for morphological operations
        min_distance: Minimum distance for watershed markers
        
    Returns:
        Dictionary containing all results and intermediate data
    """
    results = {}
    import os

    # Store identifiers for reporting
    results['file_path'] = file_path
    results['filename'] = os.path.basename(file_path)
    results['use_channels'] = use_channels
    results['region'] = region
    
    # Step 1: Load image data
    print("=== Step 1: Loading image data ===")
    arr, channel_names = load_image_data(file_path)
    results['raw_data'] = arr
    results['channel_names'] = channel_names
    
    # Step 2: Extract channels
    print("\n=== Step 2: Extracting channels ===")
    channels = extract_channels(arr)
    print(f"Channels extracted: {channels.keys()}")
    results['channels'] = channels
    
    # Step 3: Combine and scale channels
    print("\n=== Step 3: Combining and scaling channels ===")
    selected_arrays = [channels[ch] for ch in use_channels if ch in channels]

    combined_result = selected_arrays[0] #multiply_and_scale(selected_arrays)
    results['combined_result'] = combined_result
    
    # Step 4: Split regions
    print("\n=== Step 4: Splitting regions ===")
    upper, lower = split_horizontally(combined_result)
    results['regions'] = {'upper': upper, 'lower': lower}
    
    # Step 5: Select region to process
    if region == 'lower':
        image_to_process = np.squeeze(lower)
    elif region == 'upper':
        image_to_process = np.squeeze(upper)
    else:
        raise ValueError("Region must be 'upper' or 'lower'")
    
    results['image_to_process'] = image_to_process
    
    # Step 6: Thresholding
    print(f"\n=== Step 5: Thresholding ({threshold_method}) ===")
    binary_mask, threshold_value = apply_thresholding(image_to_process, threshold_method, window_size=window_size)
    results['binary_mask'] = binary_mask
    results['threshold_value'] = threshold_value
    results['threshold_method'] = threshold_method
    
    # Step 7: Post-processing
    print("\n=== Step 6: Post-processing mask ===")
    processed_mask = postprocess_mask(binary_mask, disk_radius)
    results['processed_mask'] = processed_mask
    
    # Step 8: Watershed segmentation
    print("\n=== Step 7: Watershed segmentation ===")
    labels, boundaries = watershed_segmentation(processed_mask, min_distance)
    results['labels'] = labels
    results['boundaries'] = boundaries
    
    # Step 9: Analysis
    print("\n=== Step 8: Object analysis ===")
    measurements = analyze_objects(labels, image_to_process)
    results['measurements'] = measurements
    
    print(f"\n=== Workflow Complete ===")
    print(f"File: {results['filename']}")
    print(f"Channels: {', '.join(results['use_channels'])}")
    print(f"Region: {results['region'].capitalize()}")
    print("-" * 25)
    print(f"Total objects found: {len(measurements)}")
    print(f"Mean area: {measurements['area'].mean():.2f}")
    print(f"Mean integrated density: {measurements['integrated_density'].mean():.2f}")
    
    return results


# Example usage as main function
def main():
    """Main function for testing the workflow."""
    import os
    
    # --- Define Paths ---
    # Use absolute paths to prevent any ambiguity.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_czi", "Input_Adult_V2", "J987_ARC _C3_slice5-Ort.czi")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
         
    # --- Run Workflow ---
    results = run_workflow(
        file_path=DATA_PATH,
        use_channels=['gal'],
        region='lower',
        threshold_method='otsu',
        window_size=41,
        disk_radius=2,
        min_distance=7
    )
    
    # --- Save Results ---
    if results:
        input_basename = os.path.splitext(os.path.basename(DATA_PATH))[0]
        
        channel_str = '_'.join(results['use_channels'])
        region_str = results['region']
        output_subdir_name = f"{channel_str}_{region_str}_{input_basename}"
        output_subdir = os.path.join(OUTPUT_DIR, output_subdir_name)
        
        save_image_with_boundaries(
            results=results,
            output_dir=output_subdir,
            filename_prefix=input_basename
        )
        
        print(f"\nAll results saved to: {output_subdir}")
    
    return results


if __name__ == "__main__":
    main()