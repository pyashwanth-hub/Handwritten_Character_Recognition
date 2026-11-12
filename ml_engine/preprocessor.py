"""
Image Preprocessing Pipeline for Handwritten Character & Word Recognition
Conforms to MNIST & EMNIST standards with adaptive stroke normalization:
- Robust RGB/RGBA grayscale conversion
- Border-based background polarity detection (light-on-dark vs dark-on-light)
- Gaussian noise suppression & speckle component filtering
- Distance-transform based adaptive stroke thickness normalization (standardizing to ~2.4px in 28x28 space)
- Bounding-box aspect-ratio preserving crop to 20x20 target frame
- 28x28 canvas embedding
- Continuous subpixel Center-of-Mass alignment via scipy.ndimage
- Peak contrast stretching to [0.0, 1.0]
- Unclear handwriting & ambiguity detection
- Connected component + vertical projection segmentation for CRNN word recognition
"""

import io
import re
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage

def data_uri_to_pil(data_uri):
    """Convert base64 data URI or raw base64 string to PIL Image."""
    if ',' in data_uri:
        data_uri = data_uri.split(',', 1)[1]
    image_bytes = base64.b64decode(data_uri)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def pil_to_data_uri(pil_img, format='PNG'):
    """Convert PIL Image to base64 data URI."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{encoded}"

def numpy_to_data_uri(arr_28x28):
    """Convert 28x28 numpy array (0..1 or 0..255) to base64 data URI for UI display."""
    if arr_28x28.max() <= 1.0:
        arr_uint8 = (arr_28x28 * 255.0).astype(np.uint8)
    else:
        arr_uint8 = arr_28x28.astype(np.uint8)
    # Upscale 28x28 to 140x140 using NEAREST so pixel grid is crisp
    img = Image.fromarray(arr_uint8).resize((140, 140), Image.Resampling.NEAREST)
    return pil_to_data_uri(img)

def preprocess_single_character(image_input, adaptive_dilation=True):
    """
    Complete preprocessor for a single handwritten character.
    Accepts PIL.Image, numpy array, or base64 string.
    Returns:
        dict: {
            'array_28x28': np.ndarray of shape (28, 28) with values in [0, 1],
            'flat': np.ndarray of shape (784,),
            'cnn_input': np.ndarray of shape (1, 28, 28, 1),
            'previews': {
                'original': str (base64 data URI),
                'threshold': str (base64 data URI),
                'centered': str (base64 data URI)
            },
            'is_blank': bool,
            'is_unclear': bool
        }
    """
    if isinstance(image_input, str):
        pil_img = data_uri_to_pil(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError("Unsupported image input type")

    pil_img = ImageOps.exif_transpose(pil_img)
    original_preview = pil_to_data_uri(pil_img.convert('RGB'))

    # 1. Grayscale Conversion & Alpha Channel Handling
    if pil_img.mode == 'RGBA':
        # Composite against black background first to inspect strokes
        bg = Image.new('RGB', pil_img.size, (0, 0, 0))
        bg.paste(pil_img, mask=pil_img.split()[3])
        gray = bg.convert('L')
    else:
        gray = pil_img.convert('L')

    raw_arr = np.array(gray, dtype=np.float32)

    # 2. Background Polarity Detection (Dark-on-Light vs Light-on-Dark)
    # Sample all 4 outer borders of the image
    border_pixels = np.concatenate([
        raw_arr[0, :], raw_arr[-1, :],
        raw_arr[:, 0], raw_arr[:, -1]
    ])
    border_median = float(np.median(border_pixels))
    mean_val = float(np.mean(raw_arr))

    if border_median > 120.0 or mean_val > 130.0:
        # Light background (e.g. white paper or white canvas) -> Invert
        arr = 255.0 - raw_arr
    else:
        # Dark background (canvas default)
        arr = raw_arr.copy()

    # 3. Noise Suppression & Contrast Stretching
    # Gentle Gaussian denoising to suppress grain
    arr = ndimage.gaussian_filter(arr, sigma=0.6)
    arr = np.clip(arr, 0.0, 255.0)

    # Check for blank image
    if np.max(arr) < 22.0 or np.sum(arr > 30.0) < 15:
        blank = np.zeros((28, 28), dtype=np.float32)
        blank_uri = numpy_to_data_uri(blank)
        return {
            'array_28x28': blank,
            'flat': blank.flatten(),
            'cnn_input': blank.reshape(1, 28, 28, 1),
            'previews': {
                'original': original_preview,
                'threshold': blank_uri,
                'centered': blank_uri
            },
            'is_blank': True,
            'is_unclear': False
        }

    # Dynamic Thresholding to separate foreground strokes
    pos_vals = arr[arr > 12.0]
    thresh_base = np.percentile(pos_vals, 20) if len(pos_vals) > 0 else 25.0
    thresh_val = float(np.clip(thresh_base, 20.0, 90.0))
    binary_mask = (arr > thresh_val).astype(np.uint8)

    # Speckle noise filter: remove tiny isolated components
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features > 1:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
        # Keep components with at least 15 pixels
        keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= 15]
        binary_mask = np.isin(labeled_mask, keep_labels).astype(np.uint8)

    threshold_preview = pil_to_data_uri(Image.fromarray(binary_mask * 255).resize((140, 140), Image.Resampling.NEAREST))

    # 4. Find Foreground Bounding Box
    rows = np.any(binary_mask > 0, axis=1)
    cols = np.any(binary_mask > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        blank = np.zeros((28, 28), dtype=np.float32)
        blank_uri = numpy_to_data_uri(blank)
        return {
            'array_28x28': blank,
            'flat': blank.flatten(),
            'cnn_input': blank.reshape(1, 28, 28, 1),
            'previews': {
                'original': original_preview,
                'threshold': blank_uri,
                'centered': blank_uri
            },
            'is_blank': True,
            'is_unclear': False
        }

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop_h = rmax - rmin + 1
    crop_w = cmax - cmin + 1

    # Check for unclear / chaotic smudge or tiny spec
    total_stroke_pixels = np.sum(binary_mask)
    if total_stroke_pixels < 15 or crop_h < 8 or crop_w < 2:
        blank = np.zeros((28, 28), dtype=np.float32)
        return {
            'array_28x28': blank,
            'flat': blank.flatten(),
            'cnn_input': blank.reshape(1, 28, 28, 1),
            'previews': {
                'original': original_preview,
                'threshold': threshold_preview,
                'centered': numpy_to_data_uri(blank)
            },
            'is_blank': False,
            'is_unclear': True
        }

    # 5. Adaptive Stroke Width Normalization
    # Standard MNIST/EMNIST strokes have ~2.4px width on a 28x28 canvas.
    cropped_arr = arr[rmin:rmax+1, cmin:cmax+1]
    cropped_bin = binary_mask[rmin:rmax+1, cmin:cmax+1]
    cropped_pil = Image.fromarray(np.clip(cropped_arr, 0, 255).astype(np.uint8))

    scale = 20.0 / max(crop_h, crop_w)

    if adaptive_dilation and max(crop_h, crop_w) >= 28:
        # Compute distance transform to find stroke radius
        dist = ndimage.distance_transform_edt(cropped_bin)
        max_dist = float(np.max(dist)) if dist.size > 0 else 1.0
        est_stroke_width = max(1.0, max_dist * 2.0)
        projected_stroke = est_stroke_width * scale

        # If projected stroke width would shrink below 2.2px on 28x28, dilate proportionally
        if projected_stroke < 2.2:
            kernel_size = int(round((2.3 / scale) - est_stroke_width))
            kernel_size = max(1, min(9, kernel_size))
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size >= 3:
                cropped_pil = cropped_pil.filter(ImageFilter.MaxFilter(kernel_size))
        elif projected_stroke > 3.8:
            # If overly thick, slightly erode
            cropped_pil = cropped_pil.filter(ImageFilter.MinFilter(3))

    # 6. Fit inside 20x20 Target Box (MNIST Standard)
    nw = max(1, int(round(crop_w * scale)))
    nh = max(1, int(round(crop_h * scale)))

    # For tall skinny characters ('1', 'I'), preserve their thin width
    aspect = crop_w / max(1, crop_h)
    if aspect < 0.28:
        nw = max(4, nw)

    resized = cropped_pil.resize((nw, nh), Image.Resampling.BILINEAR)

    # 7. Embed in 28x28 Canvas
    canvas_28 = Image.new('L', (28, 28), 0)
    paste_x = (28 - nw) // 2
    paste_y = (28 - nh) // 2
    canvas_28.paste(resized, (paste_x, paste_y))

    arr_28 = np.array(canvas_28, dtype=np.float32)

    # 8. Continuous Subpixel Center-of-Mass Alignment (Exact LeCun Standard)
    cy, cx = ndimage.center_of_mass(arr_28)
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = float(np.clip(14.0 - cx, -3.5, 3.5))
        shift_y = float(np.clip(14.0 - cy, -3.5, 3.5))
        arr_28 = ndimage.shift(arr_28, (shift_y, shift_x), order=1, mode='constant', cval=0.0)

    # 9. Contrast Normalization to [0.0, 1.0] with Peak Stretching
    peak = float(np.max(arr_28))
    if peak > 15.0:
        arr_28 = np.clip(arr_28 / peak, 0.0, 1.0)
    else:
        arr_28 = np.zeros((28, 28), dtype=np.float32)

    centered_preview = numpy_to_data_uri(arr_28)

    return {
        'array_28x28': arr_28,
        'flat': arr_28.flatten(),
        'cnn_input': arr_28.reshape(1, 28, 28, 1),
        'previews': {
            'original': original_preview,
            'threshold': threshold_preview,
            'centered': centered_preview
        },
        'is_blank': False,
        'is_unclear': False
    }

def segment_characters_in_word(image_input):
    """
    Advanced multi-character segmentation for CRNN word & sentence recognition.
    Combines connected component labeling and vertical projection profile
    valley splitting for connected cursive handwriting.
    """
    if isinstance(image_input, str):
        pil_img = data_uri_to_pil(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        pil_img = Image.fromarray(image_input)

    gray = pil_img.convert('L')
    raw_arr = np.array(gray, dtype=np.float32)

    # Polarity check
    borders = np.concatenate([raw_arr[0, :], raw_arr[-1, :], raw_arr[:, 0], raw_arr[:, -1]])
    if np.median(borders) > 120.0 or np.mean(raw_arr) > 130.0:
        arr = 255.0 - raw_arr
    else:
        arr = raw_arr.copy()

    # Denoise & threshold
    arr = ndimage.gaussian_filter(arr, sigma=0.6)
    thresh = max(25.0, np.percentile(arr[arr > 12], 25)) if np.any(arr > 12) else 30.0
    bin_arr = (arr > thresh).astype(np.uint8)

    # Label connected components
    labeled, num_features = ndimage.label(bin_arr)
    objects = ndimage.find_objects(labeled)

    raw_boxes = []
    for obj in objects:
        if obj is None:
            continue
        slice_y, slice_x = obj
        h = slice_y.stop - slice_y.start
        w = slice_x.stop - slice_x.start
        if h < 8 or w < 3:
            continue
        raw_boxes.append([slice_x.start, slice_y.start, slice_x.stop, slice_y.stop])

    # If cursive letters are connected horizontally, split wide components using vertical projection
    boxes = []
    for x1, y1, x2, y2 in raw_boxes:
        w = x2 - x1
        h = y2 - y1
        if w > 1.8 * h and w > 45:
            # Component is much wider than tall -> likely connected characters
            sub_bin = bin_arr[y1:y2, x1:x2]
            v_proj = np.sum(sub_bin, axis=0)
            # Find valleys in middle 80%
            start_m, end_m = int(w * 0.25), int(w * 0.75)
            if end_m > start_m:
                valley_rel = start_m + int(np.argmin(v_proj[start_m:end_m]))
                valley_x = x1 + valley_rel
                boxes.append([x1, y1, valley_x, y2])
                boxes.append([valley_x, y1, x2, y2])
            else:
                boxes.append([x1, y1, x2, y2])
        else:
            boxes.append([x1, y1, x2, y2])

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    if not boxes:
        return [preprocess_single_character(pil_img)]

    characters = []
    pad = 3
    for x1, y1, x2, y2 in boxes:
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(pil_img.width, x2 + pad)
        cy2 = min(pil_img.height, y2 + pad)

        char_crop = pil_img.crop((cx1, cy1, cx2, cy2))
        char_data = preprocess_single_character(char_crop)
        if not char_data.get('is_blank', False):
            char_data['bbox'] = [cx1, cy1, cx2, cy2]
            characters.append(char_data)

    return characters if characters else [preprocess_single_character(pil_img)]
