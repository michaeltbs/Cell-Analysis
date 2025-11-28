import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List
import warnings

try:
    import tifffile as tiff
except ImportError:
    tiff = None

from skimage import io as skio
from skimage import img_as_ubyte, util, color, morphology, transform
from skimage.filters import threshold_otsu
from matplotlib import colors as mcolors

def ensure_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    """
    Convert array to specified dtype (uint8 or uint16) with proper scaling.
    """
    if dtype == "uint16":
        if arr.dtype == np.uint16:
            return arr
        if arr.dtype == np.uint8:
            return (arr.astype(np.float32) / 255.0 * 65535.0 + 0.5).astype(np.uint16)
        m = float(arr.max()) or 1.0
        return (arr.astype(np.float32) / m * 65535.0 + 0.5).astype(np.uint16)
    
    # default to uint8
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    
    m = float(arr.max()) or 1.0
    return img_as_ubyte(arr.astype(np.float32) / m)

def normalize_stack(cyx: np.ndarray, as_uint8: bool = True) -> np.ndarray:
    """
    Per-channel percentile normalization (2% - 99.8%).
    Input expected as (C, Y, X).
    """
    out = []
    for i in range(cyx.shape[0]):
        plane = cyx[i].astype(np.float32)
        lo, hi = np.percentile(plane, (2, 99.8))
        if hi <= lo:
            scaled = np.zeros_like(plane, dtype=np.uint8)
        else:
            scaled = np.clip((plane - lo) / (hi - lo), 0, 1) * 255.0
        out.append(scaled.astype(np.uint8))
    
    stack = np.stack(out, axis=0)
    if not as_uint8:
        return stack.astype(np.uint16) * 257
    return stack

def read_image(path: Union[str, Path]) -> np.ndarray:
    """
    Robust image reader handling TIFF and other formats.
    Returns numpy array.
    """
    path_obj = Path(path)
    try:
        if tiff and path_obj.suffix.lower() in (".tif", ".tiff"):
            arr = tiff.imread(str(path_obj))
        else:
            arr = skio.imread(str(path_obj))
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path_obj.name}: {e}")
    return arr

def read_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Read an image and ensure it is a binary mask (uint8, 0 or 1).
    """
    arr = read_image(path)
    arr = util.img_as_float32(arr)
    
    if arr.ndim == 3:
        # collapse RGB/RGBA to grayscale
        arr = color.rgb2gray(arr) if arr.shape[-1] >= 3 else arr[..., 0]
        
    # binarize
    if arr.dtype != np.uint8:
        thr = threshold_otsu(arr) if np.any(arr > 0) else 0.0
        arr = (arr > thr).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
        
    return arr

def parse_color(cstr: str) -> Tuple[float, float, float]:
    """
    Parse a color string (name or hex) to (r, g, b) float tuple.
    """
    try:
        return mcolors.to_rgb(cstr)
    except Exception:
        return (1.0, 1.0, 1.0)

def prepare_display(arr: Optional[np.ndarray]) -> np.ndarray:
    """
    Normalize image data to 0.0-1.0 float range for display/processing.
    Handles 2D (grayscale) and 3D (multichannel/RGB) arrays.
    """
    if arr is None:
        return np.array([])
    data = np.asarray(arr)
    if data.size == 0:
        return data
    
    if data.ndim == 2:
        data = data.astype(np.float32, copy=False)
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            data = (data - lo) / (hi - lo)
        elif hi not in (0.0, np.nan):
            data = data / hi if hi != 0 else data
        return np.clip(data, 0.0, 1.0)
    
    data = data.astype(np.float32, copy=False)
    if data.shape[-1] > 3:
        data = data[..., :3]
    
    for c in range(data.shape[-1]):
        chan = data[..., c]
        lo, hi = float(np.nanmin(chan)), float(np.nanmax(chan))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            data[..., c] = (chan - lo) / (hi - lo)
        elif hi not in (0.0, np.nan):
            data[..., c] = chan / hi if hi != 0 else chan
        else:
            data[..., c] = 0.0
            
    return np.clip(data, 0.0, 1.0)

def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is RGB (H, W, 3).
    """
    data = np.asarray(arr)
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=-1)
    elif data.ndim == 3:
        if data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
        elif data.shape[-1] > 3:
            data = data[..., :3]
    else:
        data = np.zeros((1, 1, 3), dtype=np.float32)
    return data.astype(np.float32, copy=False)

def mask_outline(mask_bool: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Generate an outline mask from a binary mask.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    steps = max(int(round(thickness)), 1)
    current = mask_bool.copy()
    for _ in range(steps):
        next_eroded = morphology.binary_erosion(current, morphology.disk(1))
        if not np.any(next_eroded):
            break
        current = next_eroded
    outline = mask_bool & ~current
    if not np.any(outline):
        outline = mask_bool
    return outline

def apply_colored_outline(
    base_arr: np.ndarray,
    mask_bool: np.ndarray,
    color_str: str,
    thickness: int = 1,
) -> np.ndarray:
    """
    Apply a colored outline of the mask onto the base image.
    """
    rgb = ensure_rgb(base_arr).copy()
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask_bool):
        return np.clip(rgb, 0.0, 1.0)
    
    color_rgb = np.array(parse_color(color_str), dtype=np.float32)
    outline = mask_outline(mask_bool, thickness)
    if np.any(outline):
        rgb[outline] = color_rgb
    return np.clip(rgb, 0.0, 1.0)

def apply_colored_fill(
    base_arr: np.ndarray,
    mask_bool: np.ndarray,
    color_str: str,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Apply a colored fill of the mask onto the base image with transparency.
    """
    rgb = ensure_rgb(base_arr).copy()
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask_bool):
        return np.clip(rgb, 0.0, 1.0)
    
    alpha = float(alpha)
    if alpha <= 0.0:
        return np.clip(rgb, 0.0, 1.0)
    if alpha > 1.0:
        alpha = 1.0
        
    color_rgb = np.array(parse_color(color_str), dtype=np.float32)
    rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * color_rgb
    return np.clip(rgb, 0.0, 1.0)

def resize_bool(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a boolean mask to target shape.
    """
    arr = np.asarray(mask) > 0
    if arr.shape == target_shape:
        return arr
    resized = transform.resize(
        arr.astype(np.float32),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized > 0.5

def extract_background(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Extract grayscale background from an image (RGB or grayscale).
    """
    if arr is None:
        return None
    data = np.asarray(arr)
    if data.ndim == 3 and data.shape[-1] >= 3:
        data = util.img_as_float32(data)
        return color.rgb2gray(data)
    return data
