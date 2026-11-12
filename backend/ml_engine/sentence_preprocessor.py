"""
Sentence and Paragraph Preprocessor for Handwritten Text Recognition.
Handles:
- Polarity auto-detection (Light-on-Dark canvas vs Dark-on-Light paper uploads)
- Multi-line paragraph segmentation via Horizontal Projection Profile (HPP)
- Aspect-ratio preserving line normalization to standard height 32
- Bounding box extraction for words and lines
- Visual annotation generator for UI display
"""

import io
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from scipy import ndimage


def data_uri_to_pil(data_uri):
    """Decode base64 data URI to PIL image."""
    if ',' in data_uri:
        data_uri = data_uri.split(',', 1)[1]
    image_bytes = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(image_bytes))


def pil_to_data_uri(pil_img, format='PNG'):
    """Encode PIL image to base64 data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return f"data:image/{format.lower()};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def otsu_threshold(gray_arr):
    """Calculates Otsu's optimal binarization threshold."""
    hist, _ = np.histogram(gray_arr.ravel(), bins=256, range=(0, 256))
    total = gray_arr.size
    sum_total = np.dot(np.arange(256), hist)
    curr_max, threshold = 0, 0
    sum_b, w_b = 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        v = w_b * w_f * ((m_b - m_f) ** 2)
        if v > curr_max:
            curr_max, threshold = v, t
    return threshold


def extract_universal_ink(gray_arr):
    """
    Universally extracts dark handwritten ink across all photographic conditions:
    1. Dark canvas with light strokes (digital canvas)
    2. Paper card / document photographed on a darker surface / table
    3. White paper scan / photo filling the entire frame
    """
    H, W = gray_arr.shape
    border_px = np.concatenate([gray_arr[0, :], gray_arr[-1, :], gray_arr[:, 0], gray_arr[:, -1]])
    border_med = float(np.median(border_px))
    overall_mean = float(np.mean(gray_arr))
    p85 = float(np.percentile(gray_arr, 85))

    # Case A: Digital canvas (dark background, bright drawing strokes)
    if border_med < 40.0 and overall_mean < 60.0 and np.sum(gray_arr > 80) < 0.18 * (H * W):
        stroke_arr = gray_arr.astype(np.float32)
        bin_mask = (gray_arr > 25).astype(np.uint8)
        return stroke_arr, bin_mask

    def _clean_edge_artifacts(ink_arr):
        if H < 40 or W < 40:
            return ink_arr
        lab_c, num_c = ndimage.label(ink_arr)
        if num_c == 0:
            return ink_arr
        objs = ndimage.find_objects(lab_c)
        corner_labels = {
            int(lab_c[ry, rx])
            for ry in (0, H - 1) for rx in (0, W - 1)
            if lab_c[ry, rx] > 0
        }
        max_corner_area = max(400, int(0.025 * H * W))
        edge_strip = max(4, min(8, int(round(min(H, W) * 0.008))))
        cleaned = ink_arr.copy()
        for lbl_idx, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            sy, sx = sl
            comp_h = sy.stop - sy.start
            comp_w = sx.stop - sx.start
            touches_y = (sy.start == 0) or (sy.stop == H)
            touches_x = (sx.start == 0) or (sx.stop == W)
            mask_sub = (lab_c[sl] == lbl_idx)
            if lbl_idx in corner_labels and int(np.sum(mask_sub)) < max_corner_area:
                cleaned[sl][mask_sub] = False
            elif (touches_x and comp_w <= edge_strip) or (touches_y and comp_h <= edge_strip):
                cleaned[sl][mask_sub] = False
        return cleaned

    # Case B: Photo of a paper document / card on a darker desk / table
    t = otsu_threshold(gray_arr)
    p95 = float(np.percentile(gray_arr, 95))
    if border_med < 90.0 and p95 > 98.0:
        dark_mask = (gray_arr < t)
        lab, num = ndimage.label(dark_mask)
        border_labels = set(np.unique(np.concatenate([lab[0,:], lab[-1,:], lab[:,0], lab[:,-1]])))
        desk_mask = np.zeros((H, W), dtype=bool)
        for b in border_labels:
            if b > 0:
                desk_mask |= (lab == b)
        # Expand desk mask proportionally to cover paper edge shadow halo cleanly
        edge_pad = max(6, min(16, int(round(min(H, W) * 0.028))))
        desk_dil = ndimage.binary_dilation(desk_mask, iterations=edge_pad)
        paper_mask = ~desk_dil
        if np.sum(paper_mask) > 100:
            # Nearest-paper border extrapolation avoids fake halos along dim card borders
            _, (iy, ix) = ndimage.distance_transform_edt(~paper_mask, return_indices=True)
            filled_gray = gray_arr[iy, ix].astype(np.float32)
            sigma = max(14.0, min(28.0, min(H, W) * 0.045))
            bg = ndimage.gaussian_filter(filled_gray, sigma=sigma)
            diff = np.maximum(0.0, bg - filled_gray)
            p75_paper = float(np.percentile(gray_arr[paper_mask], 75))
            ink = paper_mask & (diff > 14.0) & (filled_gray < (p75_paper - 10.0))
            ink = _clean_edge_artifacts(ink)
            ink_dil = ndimage.binary_dilation(ink, iterations=1) & paper_mask
            stroke_arr = np.where(ink_dil, diff, 0.0).astype(np.float32)
            bin_mask = ink.astype(np.uint8)
            return stroke_arr, bin_mask

        ink = _clean_edge_artifacts(dark_mask & (~desk_mask))
        stroke_arr = ink.astype(np.float32) * 255.0
        bin_mask = ink.astype(np.uint8)
        return stroke_arr, bin_mask

    # Case C: White paper scan or photo filling the entire image frame
    sigma = max(14.0, min(28.0, min(H, W) * 0.045))
    bg = ndimage.gaussian_filter(gray_arr.astype(np.float32), sigma=sigma)
    diff = np.maximum(0, bg - gray_arr.astype(np.float32))
    p75 = float(np.percentile(gray_arr, 75))
    ink = (diff > 14.0) & (gray_arr < (p75 - 10.0))
    ink = _clean_edge_artifacts(ink)
    ink_dil = ndimage.binary_dilation(ink, iterations=1)
    stroke_arr = np.where(ink_dil, diff, 0.0).astype(np.float32)
    bin_mask = ink.astype(np.uint8)
    return stroke_arr, bin_mask


def _normalize_line_32(crop_pil, w_src, h_src):
    """Resize a line crop to 32px height preserving true aspect ratio, padding short lines symmetrically."""
    target_h = 32
    scale = target_h / float(max(1, h_src))
    raw_w = max(8, int(round(w_src * scale)))
    resized_line = crop_pil.resize((raw_w, target_h), Image.Resampling.BILINEAR)

    # Pad to >= 80px wide instead of stretching narrow single-word lines (e.g. 'I')
    total_w = max(raw_w + 16, 80)
    left_pad = (total_w - raw_w) // 2
    padded_line = Image.new('L', (total_w, target_h), 0)
    padded_line.paste(resized_line, (left_pad, 0))

    line_arr_32 = np.array(padded_line, dtype=np.float32)
    p_max = np.max(line_arr_32)
    if p_max > 15.0:
        return np.clip(line_arr_32 / p_max, 0.0, 1.0)
    return np.zeros_like(line_arr_32)


def preprocess_sentence_image(image_input):
    """
    Analyzes an input image for handwritten sentences and paragraphs.
    Returns:
        dict containing:
        - is_blank (bool)
        - is_unclear (bool)
        - lines: list of dicts with:
            - 'line_idx': int
            - 'line_arr_32': normalized (32, W) np.ndarray
            - 'bbox': [x1, y1, x2, y2] in original image space
            - 'preview': base64 data URI of line crop
        - original_preview: base64
        - bin_mask: np.ndarray
        - original_size: (width, height)
    """
    if isinstance(image_input, str):
        pil_img = data_uri_to_pil(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError("Unsupported image type")

    # Respect smartphone EXIF orientation before converting channels
    pil_img = ImageOps.exif_transpose(pil_img)

    # Alpha handling
    if pil_img.mode == 'RGBA':
        bg = Image.new('RGB', pil_img.size, (0, 0, 0))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_rgb = bg
    else:
        pil_rgb = pil_img.convert('RGB')

    # Cap huge smartphone camera photos to <= 960px for fast, scale-invariant OCR
    if max(pil_rgb.size) > 960:
        pil_rgb = pil_rgb.copy()
        pil_rgb.thumbnail((960, 960), Image.Resampling.LANCZOS)

    orig_w, orig_h = pil_rgb.size
    gray = pil_rgb.convert('L')
    gray_uint8 = np.array(gray, dtype=np.uint8)

    # 1. Universal Background & Polarity Detection
    stroke_arr, bin_mask = extract_universal_ink(gray_uint8)

    # 2. Check for blank or unclear image
    total_stroke_px = np.sum(bin_mask)
    if total_stroke_px < 20:
        return {
            'is_blank': True,
            'is_unclear': False,
            'lines': [],
            'original_preview': pil_to_data_uri(pil_rgb),
            'annotated_preview': pil_to_data_uri(pil_rgb),
            'original_size': (orig_w, orig_h)
        }

    if total_stroke_px < 35:
        return {
            'is_blank': False,
            'is_unclear': True,
            'lines': [],
            'original_preview': pil_to_data_uri(pil_rgb),
            'annotated_preview': pil_to_data_uri(pil_rgb),
            'original_size': (orig_w, orig_h)
        }

    # 3. Horizontal Projection Profile (HPP) for Multi-line Segmentation
    # Ignore tiny isolated speckles (<= 3px) so single-letter stems like 'I' stay unbroken
    lab_sp, num_sp = ndimage.label(bin_mask)
    if num_sp > 0:
        counts_sp = np.bincount(lab_sp.ravel())
        keep_sp = counts_sp >= 4
        keep_sp[0] = False
        hpp_mask = keep_sp[lab_sp]
    else:
        hpp_mask = bin_mask > 0
    hpp = np.sum(hpp_mask, axis=1)  # row sums

    # Use every non-empty row after speckle suppression so thin vertical stems don't fracture lines
    is_line_row = hpp >= 1

    # Morphological dilation on 1D vertical profile to connect small gaps within the same text line
    kernel_1d = np.ones(max(5, int(orig_h * 0.02)), dtype=bool)
    dilated_profile = ndimage.binary_dilation(is_line_row, structure=kernel_1d)

    labeled_lines, num_lines = ndimage.label(dilated_profile)
    slices = ndimage.find_objects(labeled_lines)

    extracted_lines = []
    line_idx = 0

    for sl in slices:
        if sl is None:
            continue
        y_slice = sl[0]
        y1 = max(0, y_slice.start - 4)
        y2 = min(orig_h, y_slice.stop + 4)
        lh = y2 - y1

        if lh < 10:  # Skip speckles
            continue

        # Find horizontal bounds for this specific line
        line_bin = bin_mask[y1:y2, :]
        vpp = np.sum(line_bin, axis=0)
        active_cols = np.where(vpp > 1)[0]
        if len(active_cols) == 0:
            continue

        x1 = max(0, active_cols[0] - 6)
        x2 = min(orig_w, active_cols[-1] + 6)
        lw = x2 - x1
        line_ink_mass = int(np.sum(line_bin[:, x1:x2]))

        if lw < 12 or line_ink_mass < 18:
            continue

        # Tight-crop vertically to actual ink rows (undoing kernel_1d vertical dilation bloat)
        row_mass = np.sum(line_bin[:, x1:x2], axis=1)
        active_rows = np.where(row_mass > 0)[0]
        if len(active_rows) > 0:
            ty1 = max(0, y1 + int(active_rows[0]) - 2)
            ty2 = min(orig_h, y1 + int(active_rows[-1]) + 3)
            if ty2 - ty1 >= 8:
                y1, y2 = ty1, ty2
                lh = y2 - y1

        # Extract line crop
        crop_stroke = stroke_arr[y1:y2, x1:x2]
        crop_pil = Image.fromarray(np.clip(crop_stroke, 0, 255).astype(np.uint8))
        line_arr_32 = _normalize_line_32(crop_pil, lw, lh)

        extracted_lines.append({
            'line_idx': line_idx,
            'line_arr_32': line_arr_32,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'preview': pil_to_data_uri(crop_pil)
        })
        line_idx += 1

    # Fallback if no clean line found
    if not extracted_lines:
        # Treat entire image as one line
        crop_pil = Image.fromarray(np.clip(stroke_arr, 0, 255).astype(np.uint8))
        line_arr_32 = _normalize_line_32(crop_pil, orig_w, orig_h)
        extracted_lines.append({
            'line_idx': 0,
            'line_arr_32': line_arr_32,
            'bbox': [0, 0, orig_w, orig_h],
            'preview': pil_to_data_uri(crop_pil)
        })

    return {
        'is_blank': False,
        'is_unclear': False,
        'lines': extracted_lines,
        'original_preview': pil_to_data_uri(pil_rgb),
        'original_size': (orig_w, orig_h),
        'bin_mask': bin_mask
    }


def draw_sentence_annotations(pil_img, words_with_bbox):
    """
    Draws aesthetic translucent bounding boxes and word tags
    on the user's handwritten sentence image.
    """
    annotated = pil_img.copy().convert('RGBA')
    overlay = Image.new('RGBA', annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Colors: vibrant neon cyan and purple
    box_color = (6, 182, 212, 60)       # Cyan translucent fill
    outline_color = (6, 182, 212, 220)  # Cyan solid border

    for item in words_with_bbox:
        bbox = item.get('bbox')
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], fill=box_color, outline=outline_color, width=2)

    combined = Image.alpha_composite(annotated, overlay)
    return pil_to_data_uri(combined.convert('RGB'))
