import os
import numpy as np
from aicspylibczi import CziFile
from skimage import io as skio
from skimage.transform import resize
from tqdm import tqdm

def convert_czi_to_target_tiff(input_dir, output_dir):
    """
    Convert CZI files to TIFF with the following specifications:
    - 8-bit (uint8)
    - 4 channels
    - 2048x2048 pixels
    - Z-projected if necessary
    """
    os.makedirs(output_dir, exist_ok=True)
    czi_files = [f for f in os.listdir(input_dir) if f.endswith('.czi')]

    for filename in tqdm(czi_files, desc=f"Converting {os.path.basename(input_dir)}"):
        input_path = os.path.join(input_dir, filename)
        output_name = filename.replace('.czi', '.tiff')
        output_path = os.path.join(output_dir, output_name)

        try:
            # Load CZI file
            czi = CziFile(input_path)
            image_data = czi.read_image()
            image_array = image_data[0] if isinstance(image_data, tuple) else image_data

            # Handle dimensions
            if czi.dims == 'HTCZYX':  # (H, T, C, Z, Y, X)
                image_array = image_array[0, 0]  # First scene and timepoint → (C, Z, Y, X)
            elif czi.dims == 'TCZYX':  # (T, C, Z, Y, X)
                image_array = image_array[0]  # First timepoint → (C, Z, Y, X)
            elif czi.dims == 'CZYX':  # (C, Z, Y, X)
                pass
            elif image_array.ndim == 3:  # (Z, Y, X)
                image_array = image_array[np.newaxis, ...]  # Add channel dimension → (1, Z, Y, X)

            # Z-projection
            if image_array.ndim == 4:  # (C, Z, Y, X)
                projected = np.max(image_array, axis=1)  # Max projection along Z → (C, Y, X)
            elif image_array.ndim == 3:  # (Z, Y, X)
                projected = np.max(image_array, axis=0)[np.newaxis, ...]  # (1, Y, X)
            else:
                projected = image_array

            # Ensure 4 channels
            if projected.shape[0] < 4:
                missing_channels = 4 - projected.shape[0]
                padding = np.zeros((missing_channels, projected.shape[1], projected.shape[2]), dtype=projected.dtype)
                projected = np.concatenate([projected, padding], axis=0)
            elif projected.shape[0] > 4:
                projected = projected[:4]  # Keep only the first 4 channels

            # Resize to 2048x2048
            if projected.shape[1] != 2048 or projected.shape[2] != 2048:
                resized_channels = [
                    resize(projected[c], (2048, 2048), preserve_range=True, anti_aliasing=True)
                    for c in range(projected.shape[0])
                ]
                projected = np.stack(resized_channels, axis=0)

            # Normalize to 8-bit
            normalized_channels = [
                ((projected[c] - projected[c].min()) / (projected[c].max() - projected[c].min()) * 255).astype(np.uint8)
                if projected[c].max() > projected[c].min() else np.zeros_like(projected[c], dtype=np.uint8)
                for c in range(projected.shape[0])
            ]
            projected = np.stack(normalized_channels, axis=0)

            # Save as multi-channel TIFF
            final_image = np.moveaxis(projected, 0, -1)  # (C, Y, X) → (Y, X, C)
            skio.imsave(output_path, final_image, check_contrast=False)
            print(f"✅ Saved: {output_path} (shape: {final_image.shape}, dtype: {final_image.dtype})")

        except Exception as e:
            print(f"❌ Error converting {filename}: {e}")
            continue

def main():
    """Convert CZI files in specified directories."""
    input_dirs = {
        "data/raw_czi/Input_old": "data/projected_tiffs/Input_old",
        "data/raw_czi/Input_adult": "data/projected_tiffs/Input_adult",
    }

    for input_dir, output_dir in input_dirs.items():
        if os.path.exists(input_dir):
            print(f"\n🔄 Converting {input_dir} → {output_dir}")
            convert_czi_to_target_tiff(input_dir, output_dir)
        else:
            print(f"⚠️ Directory not found: {input_dir}")

    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    main()
