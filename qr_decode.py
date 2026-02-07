#!/usr/bin/env python3.11
"""
QR Code Decoder - Pure Python
Usage: python3.11 qr_decode.py <image_path>
"""

import cv2
import numpy as np
import os
from itertools import combinations
from typing import List, Tuple, Optional

# Global debug output directory (None = disabled)
DEBUG_DIR = None

def _save_img(name, data, scale=None):
    """Save image to DEBUG_DIR."""
    path = os.path.join(DEBUG_DIR, name)
    if data.dtype == bool:
        cv2.imwrite(path, (data.astype(np.uint8) * 255))
    elif data.ndim == 2 and data.dtype == np.uint8 and data.max() <= 1:
        s = scale or 10
        img = ((1 - data) * 255).astype(np.uint8)
        img = cv2.resize(img, (img.shape[1]*s, img.shape[0]*s), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path, img)
    else:
        cv2.imwrite(path, data)


def _module_type_map(version):
    """Classify every module into its functional type. Returns size x size array.
    0=data, 1=finder, 2=separator, 3=timing, 4=alignment, 5=format_info, 6=version_info, 7=dark_module
    """
    size = version * 4 + 17
    t = np.zeros((size, size), dtype=np.uint8)  # 0 = data

    # Finder patterns (7x7 at three corners)
    for (r0, c0) in [(0, 0), (0, size-7), (size-7, 0)]:
        t[r0:r0+7, c0:c0+7] = 1

    # Separators (1-module white border around finders)
    # TL
    t[7, 0:8] = 2; t[0:7, 7] = 2
    # TR
    t[7, size-8:size] = 2; t[0:7, size-8] = 2
    # BL
    t[size-8, 0:8] = 2; t[size-7:size, 7] = 2

    # Timing patterns (row 6 and col 6, between separators)
    for i in range(8, size-8):
        t[6, i] = 3; t[i, 6] = 3

    # Format info
    # Horizontal: row 8, cols 0-8 and cols size-8 to size-1
    for c in range(9): t[8, c] = 5
    for c in range(size-8, size): t[8, c] = 5
    # Vertical: col 8, rows 0-8 and rows size-7 to size-1
    for r in range(9): t[r, 8] = 5
    for r in range(size-7, size): t[r, 8] = 5

    # Dark module
    t[size-8, 8] = 7

    # Version info (v >= 7)
    if version >= 7:
        t[0:6, size-11:size-8] = 6
        t[size-11:size-8, 0:6] = 6

    # Alignment patterns
    if version >= 2:
        AP_POSITIONS = {
            2:[6,18],3:[6,22],4:[6,26],5:[6,30],6:[6,34],7:[6,22,38],8:[6,24,42],9:[6,26,46],10:[6,28,50],
            11:[6,30,54],12:[6,32,58],13:[6,34,62],14:[6,26,46,66],15:[6,26,48,70],16:[6,26,50,74],
            17:[6,30,54,78],18:[6,30,56,82],19:[6,30,58,86],20:[6,34,62,90],21:[6,28,50,72,94],
            22:[6,26,50,74,98],23:[6,30,54,78,102],24:[6,28,54,80,106],25:[6,32,58,84,110],
            26:[6,30,58,86,114],27:[6,34,62,90,118],28:[6,26,50,74,98,122],29:[6,30,54,78,102,126],
            30:[6,26,52,78,104,130],31:[6,30,56,82,108,134],32:[6,34,60,86,112,138],
            33:[6,30,58,86,114,142],34:[6,34,62,90,118,146],35:[6,30,54,78,102,126,150],
            36:[6,24,50,76,102,128,154],37:[6,28,54,80,106,132,158],38:[6,32,58,84,110,136,162],
            39:[6,26,54,82,110,138,166],40:[6,30,58,86,114,142,170]
        }
        positions = AP_POSITIONS.get(version, [])
        for ar in positions:
            for ac in positions:
                if ar <= 8 and ac <= 8: continue
                if ar <= 8 and ac >= size-9: continue
                if ar >= size-9 and ac <= 8: continue
                t[ar-2:ar+3, ac-2:ac+3] = 4

    return t


def _draw_colored_matrix(matrix, version):
    """Draw QR matrix with different colors for each functional region."""
    size = matrix.shape[0]
    scale = 20
    tmap = _module_type_map(version)

    # BGR colors for each type: [data, finder, separator, timing, alignment, format, version, dark]
    # For data modules: black/white as normal
    # For function modules: tint with the type color
    COLORS = {
        1: (0, 0, 200),     # finder: red
        2: (0, 140, 255),   # separator: orange
        3: (0, 200, 200),   # timing: yellow
        4: (200, 100, 0),   # alignment: blue
        5: (200, 0, 200),   # format info: magenta
        6: (200, 200, 0),   # version info: cyan
        7: (100, 100, 100), # dark module: gray
    }

    vis = np.zeros((size * scale, size * scale, 3), dtype=np.uint8)
    for r in range(size):
        for c in range(size):
            y0, y1 = r * scale, (r + 1) * scale
            x0, x1 = c * scale, (c + 1) * scale
            mt = tmap[r, c]
            if mt == 0:
                # Data module: normal black/white
                val = 255 if matrix[r, c] == 0 else 0
                vis[y0:y1, x0:x1] = val
            else:
                color = COLORS.get(mt, (128, 128, 128))
                if matrix[r, c] == 1:
                    # Dark module: use full color
                    vis[y0:y1, x0:x1] = color
                else:
                    # Light module: use lighter tint
                    vis[y0:y1, x0:x1] = tuple(min(255, int(v * 0.4 + 255 * 0.6)) for v in color)
            # Grid line
            vis[y0, x0:x1] = (60, 60, 60)
            vis[y0:y1, x0] = (60, 60, 60)

    # Draw actual zigzag traversal path through data modules
    half = scale // 2
    path = []  # list of (row, col) in reading order
    col, up = size - 1, True
    while col >= 0:
        if col == 6:
            col -= 1
            continue
        for row in (range(size-1, -1, -1) if up else range(size)):
            if is_data_module(row, col, size):
                path.append((row, col))
            if col > 0 and is_data_module(row, col-1, size):
                path.append((row, col-1))
        col -= 2
        up = not up

    # Draw lines connecting consecutive data modules
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        x1, y1 = c1 * scale + half, r1 * scale + half
        x2, y2 = c2 * scale + half, r2 * scale + half
        # Color gradient: green at start -> red at end
        t = i / max(len(path) - 1, 1)
        color = (0, int(200 * (1 - t)), int(200 * t))  # BGR: green -> red
        cv2.line(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Draw start/end markers
    if path:
        sr, sc = path[0]
        cv2.circle(vis, (sc * scale + half, sr * scale + half), 4, (0, 255, 0), -1)
        er, ec = path[-1]
        cv2.circle(vis, (ec * scale + half, er * scale + half), 4, (0, 0, 255), -1)

    return vis


def save_debug_all(image, patterns, corners, warped_gray, warped_binary, warped_color,
                   matrix, erasure_mask, unmasked, version, ec_name, mask, codewords,
                   erasure_cws, rs_info, result):
    """Save all intermediate results to DEBUG_DIR in one go."""
    if DEBUG_DIR is None:
        return

    # 1. Finder patterns on original image
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, p in enumerate(patterns):
        cx, cy = int(p['center'][0]), int(p['center'][1])
        cv2.drawContours(vis, [p['contour']], -1, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if p.get('corners') is not None:
            for px, py in p['corners']:
                cv2.circle(vis, (int(px), int(py)), 4, (255, 0, 0), -1)
    # Draw QR boundary
    pts = corners.astype(int)
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 255), 2)
    labels = ['TL', 'TR', 'BR', 'BL']
    for i in range(4):
        cv2.circle(vis, tuple(pts[i]), 8, (0, 255, 255), -1)
        cv2.putText(vis, labels[i], (pts[i][0]+10, pts[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    # 1: detected finder patterns + QR boundary
    _save_img("1_detected.png", vis)
    # 2: warped (gray + binary side-by-side, or with color if available)
    gray3 = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    bin3 = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
    panels = [gray3, bin3] + ([warped_color] if warped_color is not None else [])
    _save_img("2_warped.png", np.hstack(panels))
    # 3: sampled matrix with color-coded function regions
    _save_img("3_matrix.png", _draw_colored_matrix(matrix, version))
    # 4: erasures (only if logo detected)
    n = 4
    if erasure_mask is not None and np.any(erasure_mask):
        size = matrix.shape[0]
        scale = 10
        vis = ((1 - matrix) * 255).astype(np.uint8)
        vis = cv2.resize(vis, (size*scale, size*scale), interpolation=cv2.INTER_NEAREST)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for r in range(size):
            for c in range(size):
                if erasure_mask[r, c]:
                    cv2.rectangle(vis, (c*scale, r*scale), ((c+1)*scale-1, (r+1)*scale-1), (0, 0, 200), -1)
        _save_img(f"{n}_erasures.png", vis); n += 1
    # unmasked: data bits after XOR mask removal
    _save_img(f"{n}_unmasked.png", unmasked); n += 1
    # RS correction details
    rs_lines = []
    for b in rs_info:
        rs_lines.append(f"  Block {b['block']}: {b['data_len']}+{b['ec_len']} bytes, "
                        f"{b['errors']} errors, {b['erasures']} erasures -> {b['status']}")
    rs_text = '\n'.join(rs_lines)
    with open(os.path.join(DEBUG_DIR, f"{n}_rs.txt"), 'w') as f:
        f.write(rs_text + '\n')
    n += 1
    # info + result
    size = version * 4 + 17
    info = (f"Version: {version}\nEC level: {ec_name}\nMask: {mask}\n"
            f"Size: {size}x{size}\nCodewords: {len(codewords)}\n"
            f"Erasure codewords: {len(erasure_cws) if erasure_cws else 0}\n"
            f"\nRS correction:\n{rs_text}\n"
            f"\nResult:\n{result}\n")
    with open(os.path.join(DEBUG_DIR, f"{n}_info.txt"), 'w') as f:
        f.write(info)

# ============================================================================
# GALOIS FIELD & REED-SOLOMON
# ============================================================================

class GF:
    """Galois Field GF(2^8) for Reed-Solomon."""
    def __init__(self):
        self.exp, self.log = [0]*512, [0]*256
        x = 1
        for i in range(255):
            self.exp[i], self.log[x] = x, i
            x = (x << 1) ^ 0x11D if x & 0x80 else x << 1
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def mul(self, a, b):
        return 0 if a == 0 or b == 0 else self.exp[self.log[a] + self.log[b]]

    def div(self, a, b):
        return 0 if a == 0 else self.exp[(self.log[a] - self.log[b]) % 255]

    def inv(self, a):
        return self.exp[255 - self.log[a]]


class ReedSolomon:
    """Reed-Solomon error correction with erasure support."""
    def __init__(self, nsym):
        self.nsym, self.gf = nsym, GF()

    def _poly_mul(self, p1, p2):
        r = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                r[i + j] ^= self.gf.mul(c1, c2)
        return r

    def _poly_eval(self, p, x):
        r = 0
        for c in p:
            r = self.gf.mul(r, x) ^ c
        return r

    def _syndromes(self, msg):
        return [self._poly_eval(msg, self.gf.exp[i]) for i in range(self.nsym)]

    def _berlekamp_massey(self, syndromes):
        """Berlekamp-Massey algorithm to find error locator polynomial."""
        n = len(syndromes)
        C = [1]
        B = [1]
        L = 0
        m = 1
        b = 1

        for n_i in range(n):
            d = syndromes[n_i]
            for i in range(1, L + 1):
                if i < len(C):
                    d ^= self.gf.mul(C[i], syndromes[n_i - i])

            if d == 0:
                m += 1
            elif 2 * L <= n_i:
                T = C.copy()
                C = C + [0] * (m - (len(C) - len(B)))
                for i in range(len(B)):
                    C[i + m] ^= self.gf.mul(self.gf.div(d, b), B[i])
                L = n_i + 1 - L
                B = T
                b = d
                m = 1
            else:
                C = C + [0] * max(0, len(B) + m - len(C))
                for i in range(len(B)):
                    C[i + m] ^= self.gf.mul(self.gf.div(d, b), B[i])
                m += 1

        return C

    def _forney_syndromes(self, syndromes, erasure_pos, n):
        """Compute Forney syndromes for erasure+error correction."""
        fsynd = list(syndromes)
        for pos in erasure_pos:
            x = self.gf.exp[n - 1 - pos]
            for i in range(len(fsynd) - 1):
                fsynd[i] = self.gf.mul(fsynd[i], x) ^ fsynd[i + 1]
        return fsynd[:-len(erasure_pos)] if erasure_pos else fsynd

    def _find_errors(self, err_loc, n):
        return [n - 1 - i for i in range(n) if self._poly_eval(err_loc, self.gf.exp[i]) == 0]

    def _correct(self, msg, synd, pos):
        if not pos: return msg
        # Build standard error locator Λ(x) = Π(1 + X_j*x), descending order
        err_loc = [1]
        for p in pos:
            err_loc = self._poly_mul(err_loc, [self.gf.exp[len(msg) - 1 - p], 1])
        omega = self._poly_mul(synd[::-1], err_loc)[-(self.nsym):]
        # Formal derivative in GF(2^8): d/dx(a*x^k) = a*x^{k-1} if k odd, 0 if k even
        n = len(err_loc) - 1
        deriv = [err_loc[i] if (n - i) % 2 == 1 else 0 for i in range(n)] or [0]
        msg = list(msg)
        for p in pos:
            Xi = self.gf.exp[len(msg) - 1 - p]
            d = self._poly_eval(deriv, self.gf.inv(Xi))
            if d != 0:
                msg[p] ^= self.gf.mul(Xi, self.gf.div(self._poly_eval(omega, self.gf.inv(Xi)), d))
        return msg

    def decode(self, msg, erasure_pos=None):
        """Decode with optional erasure positions.

        With erasures: can correct 2*errors + erasures <= nsym
        Without erasures: can correct errors <= nsym/2
        """
        erasure_pos = erasure_pos or []
        synd = self._syndromes(msg)
        if max(synd) == 0: return msg[:-self.nsym]

        if erasure_pos:
            # Erasure + error correction using Forney syndromes
            if len(erasure_pos) > self.nsym:
                raise ValueError("Too many erasures")
            fsynd = self._forney_syndromes(synd, erasure_pos, len(msg))
            err_loc = self._berlekamp_massey(fsynd) if fsynd else [1]
            num_errors = len(err_loc) - 1
            if 2 * num_errors + len(erasure_pos) > self.nsym:
                raise ValueError("Too many errors+erasures")
            err_pos = self._find_errors(err_loc, len(msg)) if num_errors > 0 else []
            all_pos = list(set(erasure_pos + err_pos))
        else:
            err_loc = self._berlekamp_massey(synd)
            if len(err_loc) - 1 > self.nsym // 2: raise ValueError("Too many errors")
            all_pos = self._find_errors(err_loc, len(msg))
            if len(all_pos) != len(err_loc) - 1: raise ValueError("Cannot locate errors")

        corrected = self._correct(msg, synd, all_pos)
        if max(self._syndromes(corrected)) != 0: raise ValueError("Correction failed")
        return corrected[:-self.nsym]


# ============================================================================
# QR DETECTION
# ============================================================================

def find_finder_patterns(image):
    """Find finder patterns with their corner points.

    A finder pattern has a specific 1:1:3:1:1 ratio structure.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 10)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    def is_square(c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx): return False
        x, y, w, h = cv2.boundingRect(approx)
        return w > 10 and 0.65 < w/h < 1.35

    def get_corners(c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4:
            return None
        pts = approx.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        corners = np.zeros((4, 2), dtype=np.float32)
        corners[0] = pts[np.argmin(s)]  # TL
        corners[2] = pts[np.argmax(s)]  # BR
        corners[1] = pts[np.argmin(d)]  # TR
        corners[3] = pts[np.argmax(d)]  # BL
        return corners

    def center(c):
        M = cv2.moments(c)
        return (M["m10"]/M["m00"], M["m01"]/M["m00"]) if M["m00"] > 0 else (0, 0)

    def dist(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def count_children(idx):
        """Count nested children (depth)."""
        depth = 0
        child = hierarchy[idx][2]  # First child
        while child != -1:
            depth += 1
            child = hierarchy[child][2]  # Go deeper
        return depth

    # Find squares - prefer those with nested structure but accept others too
    candidates = []
    for i, cnt in enumerate(contours):
        if not is_square(cnt) or cv2.contourArea(cnt) < 500:
            continue
        depth = count_children(i)
        corners = get_corners(cnt)
        candidates.append({
            'contour': cnt,
            'center': center(cnt),
            'area': cv2.contourArea(cnt),
            'corners': corners,
            'depth': depth,
            'idx': i
        })

    # Group concentric squares (same center, different sizes)
    patterns = []
    used = set()
    for i, c1 in enumerate(candidates):
        if i in used:
            continue
        group = [c1]
        used.add(i)
        for j, c2 in enumerate(candidates):
            if j not in used and dist(c1['center'], c2['center']) < 20:
                group.append(c2)
                used.add(j)

        # Valid finder pattern: 2+ concentric squares with area ratio 2-20x
        if len(group) >= 2:
            areas = sorted([g['area'] for g in group], reverse=True)
            if 2.0 < areas[0] / areas[-1] < 25.0:
                largest = max(group, key=lambda x: x['area'])
                patterns.append(largest)

    # Remove duplicates (keep largest at each location)
    final = []
    for p in sorted(patterns, key=lambda x: -x['area']):
        if not any(dist(p['center'], f['center']) < 30 for f in final):
            final.append(p)

    return final


def identify_corners(patterns):
    """Identify TL, TR, BL corners from 3 finder patterns."""
    centers = [p['center'] for p in patterns]
    max_d, diag = 0, (0, 1)
    for i in range(3):
        for j in range(i+1, 3):
            d = (centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2
            if d > max_d: max_d, diag = d, (i, j)

    tl_idx = 3 - diag[0] - diag[1]
    p1, p2 = patterns[diag[0]], patterns[diag[1]]
    c_tl = patterns[tl_idx]['center']
    v1 = (p1['center'][0] - c_tl[0], p1['center'][1] - c_tl[1])
    v2 = (p2['center'][0] - c_tl[0], p2['center'][1] - c_tl[1])
    if v1[0] * v2[1] - v1[1] * v2[0] > 0:
        return patterns[tl_idx], p1, p2
    return patterns[tl_idx], p2, p1


def group_finder_patterns(patterns):
    """Group finder patterns that belong to the same QR code.

    Returns list of (tl, tr, bl) tuples for each detected QR code.
    """
    if len(patterns) < 3:
        return []

    def dist(p1, p2):
        return np.sqrt((p1['center'][0]-p2['center'][0])**2 +
                       (p1['center'][1]-p2['center'][1])**2)

    def is_valid_qr_geometry(p1, p2, p3):
        """Check if 3 patterns form valid QR geometry (right angle at TL)."""
        centers = [p1['center'], p2['center'], p3['center']]

        # Find the pattern at the right angle (TL)
        for i in range(3):
            c = centers[i]
            others = [centers[j] for j in range(3) if j != i]
            v1 = (others[0][0] - c[0], others[0][1] - c[1])
            v2 = (others[1][0] - c[0], others[1][1] - c[1])

            # Check angle (should be close to 90 degrees)
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if len1 == 0 or len2 == 0:
                continue
            dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
            angle = np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi

            # Check that sides are similar length (within 3x ratio for perspective)
            ratio = max(len1, len2) / min(len1, len2)

            if 60 < angle < 120 and ratio < 3:
                return True
        return False

    # Try all combinations of 3 patterns
    valid_groups = []
    for combo in combinations(range(len(patterns)), 3):
        p1, p2, p3 = patterns[combo[0]], patterns[combo[1]], patterns[combo[2]]
        if is_valid_qr_geometry(p1, p2, p3):
            # Check that pattern sizes are similar (within 4x for perspective)
            sizes = [np.sqrt(p['area']) for p in [p1, p2, p3]]
            if max(sizes) / min(sizes) < 4:
                valid_groups.append((p1, p2, p3))

    # Convert to (tl, tr, bl) format
    result = []
    for group in valid_groups:
        try:
            tl, tr, bl = identify_corners(list(group))
            result.append((tl, tr, bl))
        except:
            pass

    return result


def find_alignment_pattern(image, expected_center, search_radius, module_size):
    """Find alignment pattern near expected position.

    The alignment pattern is a 5x5 module pattern:
    █████
    █░░░█
    █░█░█
    █░░░█
    █████

    Returns: (x, y) center of alignment pattern, or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    x_c, y_c = int(expected_center[0]), int(expected_center[1])
    r = int(search_radius)

    # Extract search region
    x1, y1 = max(0, x_c - r), max(0, y_c - r)
    x2, y2 = min(gray.shape[1], x_c + r), min(gray.shape[0], y_c + r)
    region = gray[y1:y2, x1:x2]

    if region.size == 0:
        return None

    # Threshold
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours - looking for small dark squares
    contours, hierarchy = cv2.findContours(255 - binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_dist = float('inf')

    # Expected size of alignment pattern center (1x1 module)
    expected_center_area = module_size * module_size
    # Stricter distance threshold - must be within 2 modules of expected
    max_dist = 2 * module_size

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # Look for the center black dot (about 1 module squared)
        if expected_center_area * 0.3 < area < expected_center_area * 3:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00'] + x1
                cy = M['m01'] / M['m00'] + y1

                # Verify it's very close to expected position
                dist = np.sqrt((cx - expected_center[0])**2 + (cy - expected_center[1])**2)
                if dist > max_dist:
                    continue

                # Check if this contour has a parent (should be inside white ring)
                if hierarchy is not None and hierarchy[0][i][3] >= 0:
                    # Prefer closest to expected position
                    if dist < best_dist:
                        best_dist = dist
                        best_center = (cx, cy)

    return best_center


def get_qr_corners(tl, tr, bl, version=None, image=None):
    """Get 4 corners of QR code using finder pattern corners.

    Uses homography from finder pattern corners (not just centers) for
    accurate perspective correction. Each finder has 4 corners at known
    module positions.
    """
    tl_c = np.array(tl['center'])
    tr_c = np.array(tr['center'])
    bl_c = np.array(bl['center'])

    # Estimate version if not provided
    if version is None:
        module_size = np.sqrt(tl['area']) / 7
        dist = np.sqrt((tl_c[0]-tr_c[0])**2 + (tl_c[1]-tr_c[1])**2)
        version = max(1, min(40, round(((dist / module_size + 7) - 17) / 4)))

    size = version * 4 + 17

    # Try to use finder corners for homography
    tl_corners = tl.get('corners')
    tr_corners = tr.get('corners')
    bl_corners = bl.get('corners')

    if tl_corners is not None and tr_corners is not None and bl_corners is not None:
        # Build correspondence points from finder corners
        # Finder corners are ordered: TL, TR, BR, BL (of each finder's outer square)
        #
        # In module coordinates:
        # TL finder outer square: corners at (0,0), (7,0), (7,7), (0,7)
        # TR finder outer square: corners at (size-7,0), (size,0), (size,7), (size-7,7)
        # BL finder outer square: corners at (0,size-7), (7,size-7), (7,size), (0,size)

        img_pts = []
        mod_pts = []

        # TL finder corners
        img_pts.extend([tl_corners[0], tl_corners[1], tl_corners[2], tl_corners[3]])
        mod_pts.extend([[0, 0], [7, 0], [7, 7], [0, 7]])

        # TR finder corners
        img_pts.extend([tr_corners[0], tr_corners[1], tr_corners[2], tr_corners[3]])
        mod_pts.extend([[size-7, 0], [size, 0], [size, 7], [size-7, 7]])

        # BL finder corners
        img_pts.extend([bl_corners[0], bl_corners[1], bl_corners[2], bl_corners[3]])
        mod_pts.extend([[0, size-7], [7, size-7], [7, size], [0, size]])

        img_pts = np.array(img_pts, dtype=np.float32)
        mod_pts = np.array(mod_pts, dtype=np.float32)

        # Compute homography: module coords -> image coords
        H, mask = cv2.findHomography(mod_pts, img_pts, cv2.RANSAC, 3.0)

        if H is not None:
            # Project QR outer corners
            outer_mod = np.array([[[0, 0]], [[size, 0]], [[size, size]], [[0, size]]], dtype=np.float32)
            outer_img = cv2.perspectiveTransform(outer_mod, H)
            return outer_img.reshape(4, 2).astype(np.float32)

    # Fallback: parallelogram estimation using centers
    v_tr = tr_c - tl_c
    v_bl = bl_c - tl_c
    br_c = tl_c + v_tr + v_bl

    module_h = np.linalg.norm(v_tr) / (size - 7)
    module_v = np.linalg.norm(v_bl) / (size - 7)
    u_tr = v_tr / np.linalg.norm(v_tr)
    u_bl = v_bl / np.linalg.norm(v_bl)

    offset = 3.5
    qr_tl = tl_c - offset * module_h * u_tr - offset * module_v * u_bl
    qr_tr = tr_c + offset * module_h * u_tr - offset * module_v * u_bl
    qr_bl = bl_c - offset * module_h * u_tr + offset * module_v * u_bl
    qr_br = br_c + offset * module_h * u_tr + offset * module_v * u_bl

    return np.array([qr_tl, qr_tr, qr_br, qr_bl], dtype=np.float32)


# ============================================================================
# QR SAMPLING
# ============================================================================

def sample_matrix(image, corners, version, quick=False):
    """Sample QR matrix with optimal alignment."""
    size, warp_size = version * 4 + 17, (version * 4 + 17) * 10
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    dst = np.array([[0,0], [warp_size-1,0], [warp_size-1,warp_size-1], [0,warp_size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(gray, M, (warp_size, warp_size))
    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Warp color image for logo/erasure detection
    warped_hsv = None
    warped_color = None
    if len(image.shape) == 3:
        warped_color = cv2.warpPerspective(image, M, (warp_size, warp_size))
        warped_hsv = cv2.cvtColor(warped_color, cv2.COLOR_BGR2HSV)

    FINDER = np.array([[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],[1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,0,0,0,0,1],[1,1,1,1,1,1,1]], dtype=np.uint8)

    def sample(ox, oy, ms):
        coords_c = (ox + (np.arange(size) + 0.5) * ms).astype(int)
        coords_r = (oy + (np.arange(size) + 0.5) * ms).astype(int)
        coords_c = np.clip(coords_c, 0, warp_size-1)
        coords_r = np.clip(coords_r, 0, warp_size-1)
        m = (binary[coords_r][:, coords_c] < 128).astype(np.uint8)
        return m

    def score(m):
        s = np.sum(m[0:7, 0:7] == FINDER) + np.sum(m[0:7, size-7:size] == FINDER) + np.sum(m[size-7:size, 0:7] == FINDER)
        for i in range(8, size-8):
            if m[6, i] == (1 if i % 2 == 0 else 0): s += 1
            if m[i, 6] == (1 if i % 2 == 0 else 0): s += 1
        return s

    # Quick mode for version detection
    step = 1.0 if quick else 0.5
    best, best_s, best_p = None, 0, (0, 0, 10.0)
    for ox in np.arange(-5, 5, step):
        for oy in np.arange(-5, 5, step):
            for ms in np.arange(9.6, 10.4, 0.2 if quick else 0.1):
                m = sample(ox, oy, ms)
                s = score(m)
                if s > best_s: best, best_s, best_p = m, s, (ox, oy, ms)

    if quick:
        return best, None, None

    # Fine search
    ox, oy, ms = best_p
    for dox in np.arange(-0.5, 0.6, 0.2):
        for doy in np.arange(-0.5, 0.6, 0.2):
            for dms in np.arange(-0.15, 0.2, 0.05):
                m = sample(ox+dox, oy+doy, ms+dms)
                s = score(m)
                if s > best_s: best, best_s, best_p = m, s, (ox+dox, oy+doy, ms+dms)

    # Detect logo/erasure modules via color saturation
    erasure_mask = np.zeros((size, size), dtype=bool)
    if warped_hsv is not None:
        ox, oy, ms = best_p
        coords_c = (ox + (np.arange(size) + 0.5) * ms).astype(int)
        coords_r = (oy + (np.arange(size) + 0.5) * ms).astype(int)
        coords_c = np.clip(coords_c, 0, warp_size-1)
        coords_r = np.clip(coords_r, 0, warp_size-1)
        # Sample saturation values for all modules
        sat_values = warped_hsv[coords_r][:, coords_c, 1]
        # Use high threshold to detect strongly colored logo pixels
        high_sat = sat_values > 100
        high_count = int(np.sum(high_sat))
        if high_count > 0 and high_count < size * size // 4:
            # Found a logo region - use connected component to find its bounding box
            # then mark all modules in that box as erasures (including white gaps)
            rows, cols = np.where(high_sat)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            # Only fill bounding box if it's compact (logo-shaped, not scattered)
            bbox_area = (r_max - r_min + 1) * (c_max - c_min + 1)
            if high_count >= bbox_area * 0.15:  # At least 15% density within bbox
                erasure_mask[r_min:r_max+1, c_min:c_max+1] = True

    debug_imgs = (warped, binary, warped_color) if DEBUG_DIR and not quick else None
    return best, erasure_mask, debug_imgs


# ============================================================================
# QR DECODING
# ============================================================================

def read_format_info(matrix):
    """Read EC level and mask pattern."""
    bits = [matrix[8, c] for c in [0,1,2,3,4,5,7,8]] + [matrix[r, 8] for r in [7,5,4,3,2,1,0]]
    val = sum(b << (14-i) for i, b in enumerate(bits)) ^ 0b101010000010010
    return (val >> 13) & 0b11, (val >> 10) & 0b111


def is_data_module(r, c, size):
    """Check if position is data (not function pattern)."""
    # Finder patterns and separators (top-left, top-right, bottom-left)
    if r <= 8 and c <= 8: return False
    if r <= 8 and c >= size-8: return False
    if r >= size-8 and c <= 8: return False
    # Timing patterns
    if r == 6 or c == 6: return False
    # Format info
    if r == 8 and (c <= 8 or c >= size-8): return False
    if c == 8 and (r <= 8 or r >= size-7): return False
    if r == size-8 and c == 8: return False

    version = (size - 17) // 4

    # Version info (v >= 7): 6x3 blocks near top-right and bottom-left
    if version >= 7:
        if r < 6 and c >= size-11 and c < size-8: return False  # Top-right version
        if c < 6 and r >= size-11 and r < size-8: return False  # Bottom-left version

    # Alignment patterns - all QR codes v2+ have them
    if version >= 2:
        # Alignment pattern center positions for each version
        AP_POSITIONS = {
            2:[6,18],3:[6,22],4:[6,26],5:[6,30],6:[6,34],7:[6,22,38],8:[6,24,42],9:[6,26,46],10:[6,28,50],
            11:[6,30,54],12:[6,32,58],13:[6,34,62],14:[6,26,46,66],15:[6,26,48,70],16:[6,26,50,74],
            17:[6,30,54,78],18:[6,30,56,82],19:[6,30,58,86],20:[6,34,62,90],21:[6,28,50,72,94],
            22:[6,26,50,74,98],23:[6,30,54,78,102],24:[6,28,54,80,106],25:[6,32,58,84,110],
            26:[6,30,58,86,114],27:[6,34,62,90,118],28:[6,26,50,74,98,122],29:[6,30,54,78,102,126],
            30:[6,26,52,78,104,130],31:[6,30,56,82,108,134],32:[6,34,60,86,112,138],
            33:[6,30,58,86,114,142],34:[6,34,62,90,118,146],35:[6,30,54,78,102,126,150],
            36:[6,24,50,76,102,128,154],37:[6,28,54,80,106,132,158],38:[6,32,58,84,110,136,162],
            39:[6,26,54,82,110,138,166],40:[6,30,58,86,114,142,170]
        }
        positions = AP_POSITIONS.get(version, [])
        for ar in positions:
            for ac in positions:
                # Skip alignment patterns that overlap with finder patterns
                if ar <= 8 and ac <= 8: continue  # Top-left finder
                if ar <= 8 and ac >= size-9: continue  # Top-right finder
                if ar >= size-9 and ac <= 8: continue  # Bottom-left finder
                # Check if current position is within this alignment pattern (5x5)
                if abs(r - ar) <= 2 and abs(c - ac) <= 2:
                    return False

    return True


def unmask(matrix, mask):
    """Apply mask pattern."""
    size, result = matrix.shape[0], matrix.copy()
    for r in range(size):
        for c in range(size):
            if is_data_module(r, c, size):
                flip = [lambda r,c: (r+c)%2==0, lambda r,c: r%2==0, lambda r,c: c%3==0,
                        lambda r,c: (r+c)%3==0, lambda r,c: (r//2+c//3)%2==0,
                        lambda r,c: (r*c)%2+(r*c)%3==0, lambda r,c: ((r*c)%2+(r*c)%3)%2==0,
                        lambda r,c: ((r+c)%2+(r*c)%3)%2==0][mask](r, c)
                if flip: result[r, c] = 1 - result[r, c]
    return result


def read_codewords(matrix):
    """Read codewords in zigzag pattern."""
    size, bits, col, up = matrix.shape[0], [], matrix.shape[0] - 1, True
    while col >= 0:
        if col == 6: col -= 1; continue
        for row in (range(size-1, -1, -1) if up else range(size)):
            if is_data_module(row, col, size): bits.append(matrix[row, col])
            if col > 0 and is_data_module(row, col-1, size): bits.append(matrix[row, col-1])
        col -= 2
        up = not up
    return [sum(bits[i+j] << (7-j) for j in range(8)) for i in range(0, len(bits)-7, 8)]


def get_codeword_erasures(size, erasure_mask):
    """Map module-level erasures to codeword-level erasure indices."""
    if erasure_mask is None or not np.any(erasure_mask):
        return None

    erasure_codewords = set()
    bit_idx = 0
    col = size - 1
    up = True

    while col >= 0:
        if col == 6:
            col -= 1
            continue
        for row in (range(size-1, -1, -1) if up else range(size)):
            if is_data_module(row, col, size):
                if erasure_mask[row, col]:
                    erasure_codewords.add(bit_idx // 8)
                bit_idx += 1
            if col > 0 and is_data_module(row, col-1, size):
                if erasure_mask[row, col-1]:
                    erasure_codewords.add(bit_idx // 8)
                bit_idx += 1
        col -= 2
        up = not up

    if not erasure_codewords:
        return None
    print(f"  Logo detected: {int(np.sum(erasure_mask))} modules erased -> {len(erasure_codewords)} codewords marked")
    return erasure_codewords


def decode_with_rs(codewords, version, ec_level, erasure_codewords=None):
    """Decode with Reed-Solomon error correction."""
    # Convert format info EC level to table index
    # Format: 0=M, 1=L, 2=H, 3=Q -> Table: 0=L, 1=M, 2=Q, 3=H
    ec_table = [1, 0, 3, 2][ec_level]  # M->1, L->0, H->3, Q->2

    # Full QR block structure table: (version, ec_level) -> [(data_bytes, total_bytes), ...]
    # EC levels: 0=L, 1=M, 2=Q, 3=H
    BLOCKS = {
        (1,0):[(19,26)],(1,1):[(16,26)],(1,2):[(13,26)],(1,3):[(9,26)],
        (2,0):[(34,44)],(2,1):[(28,44)],(2,2):[(22,44)],(2,3):[(16,44)],
        (3,0):[(55,70)],(3,1):[(44,70)],(3,2):[(17,35)]*2,(3,3):[(13,35)]*2,
        (4,0):[(80,100)],(4,1):[(32,50)]*2,(4,2):[(24,50)]*2,(4,3):[(9,25)]*4,
        (5,0):[(108,134)],(5,1):[(43,67)]*2,(5,2):[(11,33)]*2+[(12,34)]*2,(5,3):[(11,33)]*2+[(12,34)]*2,
        (6,0):[(68,86)]*2,(6,1):[(27,43)]*4,(6,2):[(15,36)]*4,(6,3):[(15,36)]*4,
        (7,0):[(78,98)]*2,(7,1):[(19,44)]*4,(7,2):[(13,32)]*2+[(14,33)]*4,(7,3):[(13,32)]*4+[(14,33)]*1,
        (8,0):[(97,121)]*2,(8,1):[(22,44)]*2+[(23,45)]*2,(8,2):[(14,33)]*4+[(15,34)]*2,(8,3):[(12,30)]*4+[(13,31)]*2,
        (9,0):[(116,146)]*2,(9,1):[(22,44)]*3+[(23,45)]*2,(9,2):[(12,30)]*4+[(13,31)]*4,(9,3):[(11,28)]*4+[(12,29)]*4,
        (10,0):[(68,86)]*2+[(69,87)]*2,(10,1):[(26,54)]*4+[(27,55)]*1,(10,2):[(15,36)]*6+[(16,37)]*2,(10,3):[(12,30)]*6+[(13,31)]*2,
        (11,0):[(81,101)]*4,(11,1):[(30,58)]*1+[(31,59)]*4,(11,2):[(12,30)]*4+[(13,31)]*4,(11,3):[(12,30)]*3+[(13,31)]*8,
        (12,0):[(92,116)]*2+[(93,117)]*2,(12,1):[(22,44)]*6+[(23,45)]*2,(12,2):[(14,34)]*4+[(15,35)]*6,(12,3):[(11,28)]*7+[(12,29)]*4,
        (13,0):[(107,133)]*4,(13,1):[(33,63)]*8+[(34,64)]*1,(13,2):[(16,38)]*8+[(17,39)]*4,(13,3):[(12,30)]*12+[(13,31)]*4,
        (14,0):[(115,145)]*3+[(116,146)]*1,(14,1):[(36,68)]*4+[(37,69)]*5,(14,2):[(12,30)]*11+[(13,31)]*5,(14,3):[(11,28)]*11+[(12,29)]*5,
        (15,0):[(87,109)]*5+[(88,110)]*1,(15,1):[(36,68)]*5+[(37,69)]*5,(15,2):[(12,30)]*5+[(13,31)]*7,(15,3):[(12,30)]*11+[(13,31)]*7,
        (16,0):[(98,122)]*5+[(99,123)]*1,(16,1):[(45,85)]*7+[(46,86)]*3,(16,2):[(14,34)]*15+[(15,35)]*2,(16,3):[(12,30)]*3+[(13,31)]*13,
        (17,0):[(107,135)]*1+[(108,136)]*5,(17,1):[(46,86)]*10+[(47,87)]*1,(17,2):[(14,34)]*1+[(15,35)]*15,(17,3):[(11,28)]*2+[(12,29)]*17,
        (18,0):[(120,150)]*5+[(121,151)]*1,(18,1):[(43,81)]*9+[(44,82)]*4,(18,2):[(14,34)]*17+[(15,35)]*1,(18,3):[(11,28)]*2+[(12,29)]*19,
        (19,0):[(113,141)]*3+[(114,142)]*4,(19,1):[(44,82)]*3+[(45,83)]*11,(19,2):[(14,34)]*17+[(15,35)]*4,(19,3):[(13,32)]*9+[(14,33)]*16,
        (20,0):[(107,135)]*3+[(108,136)]*5,(20,1):[(41,77)]*3+[(42,78)]*13,(20,2):[(13,32)]*15+[(14,33)]*5,(20,3):[(12,30)]*15+[(13,31)]*10,
        (21,0):[(116,144)]*4+[(117,145)]*4,(21,1):[(42,78)]*17,(21,2):[(15,36)]*17+[(16,37)]*6,(21,3):[(12,30)]*19+[(13,31)]*6,
        (22,0):[(111,139)]*2+[(112,140)]*7,(22,1):[(46,86)]*17,(22,2):[(14,34)]*7+[(15,35)]*16,(22,3):[(13,32)]*34,
        (23,0):[(121,151)]*4+[(122,152)]*5,(23,1):[(47,87)]*4+[(48,88)]*14,(23,2):[(14,34)]*11+[(15,35)]*14,(23,3):[(13,32)]*16+[(14,33)]*14,
        (24,0):[(117,147)]*6+[(118,148)]*4,(24,1):[(45,85)]*6+[(46,86)]*14,(24,2):[(14,34)]*11+[(15,35)]*16,(24,3):[(12,30)]*30+[(13,31)]*2,
        (25,0):[(106,132)]*8+[(107,133)]*4,(25,1):[(47,87)]*8+[(48,88)]*13,(25,2):[(14,34)]*7+[(15,35)]*22,(25,3):[(12,30)]*22+[(13,31)]*13,
        (26,0):[(114,142)]*10+[(115,143)]*2,(26,1):[(46,86)]*19+[(47,87)]*4,(26,2):[(15,36)]*28+[(16,37)]*6,(26,3):[(13,32)]*33+[(14,33)]*4,
        (27,0):[(122,152)]*8+[(123,153)]*4,(27,1):[(45,85)]*22+[(46,86)]*3,(27,2):[(15,36)]*8+[(16,37)]*26,(27,3):[(12,30)]*12+[(13,31)]*28,
        (28,0):[(117,147)]*3+[(118,148)]*10,(28,1):[(45,85)]*3+[(46,86)]*23,(28,2):[(15,36)]*4+[(16,37)]*31,(28,3):[(13,32)]*11+[(14,33)]*31,
        (29,0):[(116,146)]*7+[(117,147)]*7,(29,1):[(45,85)]*21+[(46,86)]*7,(29,2):[(13,32)]*1+[(14,33)]*37,(29,3):[(12,30)]*19+[(13,31)]*26,
        (30,0):[(115,145)]*5+[(116,146)]*10,(30,1):[(47,87)]*19+[(48,88)]*10,(30,2):[(15,36)]*15+[(16,37)]*25,(30,3):[(13,32)]*23+[(14,33)]*25,
        (31,0):[(115,145)]*13+[(116,146)]*3,(31,1):[(46,86)]*2+[(47,87)]*29,(31,2):[(15,36)]*42+[(16,37)]*1,(31,3):[(13,32)]*23+[(14,33)]*28,
        (32,0):[(115,145)]*17,(32,1):[(46,86)]*10+[(47,87)]*23,(32,2):[(15,36)]*10+[(16,37)]*35,(32,3):[(13,32)]*19+[(14,33)]*35,
        (33,0):[(115,145)]*17+[(116,146)]*1,(33,1):[(46,86)]*14+[(47,87)]*21,(33,2):[(15,36)]*29+[(16,37)]*19,(33,3):[(13,32)]*11+[(14,33)]*46,
        (34,0):[(115,145)]*13+[(116,146)]*6,(34,1):[(46,86)]*14+[(47,87)]*23,(34,2):[(16,37)]*44+[(17,38)]*7,(34,3):[(13,32)]*59+[(14,33)]*1,
        (35,0):[(121,151)]*12+[(122,152)]*7,(35,1):[(47,87)]*12+[(48,88)]*26,(35,2):[(15,36)]*39+[(16,37)]*14,(35,3):[(13,32)]*22+[(14,33)]*41,
        (36,0):[(121,151)]*6+[(122,152)]*14,(36,1):[(47,87)]*6+[(48,88)]*34,(36,2):[(15,36)]*46+[(16,37)]*10,(36,3):[(13,32)]*2+[(14,33)]*64,
        (37,0):[(122,152)]*17+[(123,153)]*4,(37,1):[(46,86)]*29+[(47,87)]*14,(37,2):[(15,36)]*49+[(16,37)]*10,(37,3):[(13,32)]*24+[(14,33)]*46,
        (38,0):[(122,152)]*4+[(123,153)]*18,(38,1):[(46,86)]*13+[(47,87)]*32,(38,2):[(15,36)]*48+[(16,37)]*14,(38,3):[(13,32)]*42+[(14,33)]*32,
        (39,0):[(117,147)]*20+[(118,148)]*4,(39,1):[(47,87)]*40+[(48,88)]*7,(39,2):[(15,36)]*43+[(16,37)]*22,(39,3):[(13,32)]*10+[(14,33)]*67,
        (40,0):[(118,148)]*19+[(119,149)]*6,(40,1):[(47,87)]*18+[(48,88)]*31,(40,2):[(15,36)]*34+[(16,37)]*34,(40,3):[(13,32)]*20+[(14,33)]*61,
    }
    if (version, ec_table) not in BLOCKS:
        raise ValueError(f"Unsupported version {version} / EC {ec_table}")

    blocks = BLOCKS[(version, ec_table)]
    block_data, block_ec = [[] for _ in blocks], [[] for _ in blocks]

    idx, max_data, ec_len = 0, max(b[0] for b in blocks), blocks[0][1] - blocks[0][0]
    # Track global codeword index -> (block_idx, position_in_block) for erasure mapping
    global_to_block = {}
    for col in range(max_data):
        for i, (data_len, _) in enumerate(blocks):
            if col < data_len and idx < len(codewords):
                block_data[i].append(codewords[idx])
                global_to_block[idx] = (i, len(block_data[i]) - 1)
                idx += 1
    for col in range(ec_len):
        for i in range(len(blocks)):
            if idx < len(codewords):
                block_ec[i].append(codewords[idx])
                global_to_block[idx] = (i, blocks[i][0] + len(block_ec[i]) - 1)
                idx += 1

    # Build per-block erasure positions
    block_erasures = [[] for _ in blocks]
    if erasure_codewords:
        for g_idx in erasure_codewords:
            if g_idx in global_to_block:
                b_idx, pos = global_to_block[g_idx]
                block_erasures[b_idx].append(pos)

    rs, data = ReedSolomon(ec_len), []
    rs_info = []  # per-block RS correction details
    for i in range(len(blocks)):
        raw = block_data[i] + block_ec[i]
        try:
            erasures = block_erasures[i] if block_erasures[i] else None
            corrected = rs.decode(raw, erasure_pos=erasures)
            errors = sum(a != b for a, b in zip(raw[:blocks[i][0]], corrected))
            rs_info.append({'block': i, 'data_len': blocks[i][0], 'ec_len': ec_len,
                            'errors': errors, 'erasures': len(erasures) if erasures else 0,
                            'status': 'ok'})
            data.extend(corrected)
        except Exception as e:
            rs_info.append({'block': i, 'data_len': blocks[i][0], 'ec_len': ec_len,
                            'errors': '?', 'erasures': len(block_erasures[i]),
                            'status': f'failed: {e}'})
            data.extend(block_data[i])

    bits = [((byte >> (7-i)) & 1) for byte in data for i in range(8)]

    def read_bits(pos, n):
        return sum(bits[pos+i] << (n-1-i) for i in range(n)), pos + n

    # Character count indicator lengths by version group
    def count_bits(mode, ver):
        if ver <= 9:
            return {1: 10, 2: 9, 4: 8, 8: 8}.get(mode, 8)
        elif ver <= 26:
            return {1: 12, 2: 11, 4: 16, 8: 10}.get(mode, 16)
        else:
            return {1: 14, 2: 13, 4: 16, 8: 12}.get(mode, 16)

    ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

    result, pos = "", 0
    while pos < len(bits) - 4:
        mode, pos = read_bits(pos, 4)
        if mode == 0:  # Terminator
            break

        count, pos = read_bits(pos, count_bits(mode, version))

        if mode == 1:  # Numeric: 3 digits = 10 bits, 2 = 7 bits, 1 = 4 bits
            while count >= 3:
                val, pos = read_bits(pos, 10)
                result += f"{val:03d}"
                count -= 3
            if count == 2:
                val, pos = read_bits(pos, 7)
                result += f"{val:02d}"
            elif count == 1:
                val, pos = read_bits(pos, 4)
                result += str(val)

        elif mode == 2:  # Alphanumeric: 2 chars = 11 bits, 1 = 6 bits
            while count >= 2:
                val, pos = read_bits(pos, 11)
                result += ALNUM[val // 45] + ALNUM[val % 45]
                count -= 2
            if count == 1:
                val, pos = read_bits(pos, 6)
                result += ALNUM[val]

        elif mode == 4:  # Byte
            chars = []
            for _ in range(count):
                val, pos = read_bits(pos, 8)
                chars.append(val)
            result += bytes(chars).decode('utf-8', errors='replace')

        elif mode == 8:  # Kanji (Shift JIS)
            chars = []
            for _ in range(count):
                val, pos = read_bits(pos, 13)
                if val < 0x01F00:
                    val += 0x08140
                else:
                    val += 0x0C140
                chars.append(((val >> 8) & 0xFF, val & 0xFF))
            result += b''.join(bytes(c) for c in chars).decode('shift_jis', errors='replace')

        elif mode == 7:  # ECI (skip for now, just read the designator)
            eci, pos = read_bits(pos, 8)
            # Continue to next mode segment

        else:
            result += f"[Mode {mode}?]"
            break

    return result, rs_info


# ============================================================================
# MAIN
# ============================================================================

def try_decode_patterns(image, tl, tr, bl, patterns=None):
    """Try to decode QR using given finder patterns. Returns (success, result)."""
    try:
        # Estimate version
        dist = np.sqrt((tl['center'][0]-tr['center'][0])**2 + (tl['center'][1]-tr['center'][1])**2)
        module_size = np.sqrt(tl['area']) / 7
        version = max(1, min(40, round(((dist / module_size + 7) - 17) / 4)))

        corners = get_qr_corners(tl, tr, bl, version=version, image=image)
        matrix, erasure_mask, debug_imgs = sample_matrix(image, corners, version)

        # Quick validation: check finder patterns
        FINDER = np.array([[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],[1,0,1,1,1,0,1],
                          [1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,0,0,0,0,1],[1,1,1,1,1,1,1]])
        size = version * 4 + 17
        finder_score = (np.sum(matrix[0:7, 0:7] == FINDER) +
                       np.sum(matrix[0:7, size-7:size] == FINDER) +
                       np.sum(matrix[size-7:size, 0:7] == FINDER))
        if finder_score < 100:  # At least ~68% correct
            return False, None

        ec_level, mask = read_format_info(matrix)
        EC_NAMES = {0: 'M', 1: 'L', 2: 'H', 3: 'Q'}
        ec_name = EC_NAMES.get(ec_level, '?')
        print(f"  Version: {version}, RS level: {ec_name} (mask {mask})")
        unmasked = unmask(matrix, mask)
        codewords = read_codewords(unmasked)
        erasure_cws = get_codeword_erasures(size, erasure_mask)

        result, rs_info = decode_with_rs(codewords, version, ec_level, erasure_codewords=erasure_cws)

        # Validate result (should be printable text)
        if result and len(result) > 0 and not result.startswith('[Mode'):
            if debug_imgs:
                save_debug_all(image, patterns or [tl, tr, bl], corners,
                               debug_imgs[0], debug_imgs[1], debug_imgs[2],
                               matrix, erasure_mask, unmasked, version, ec_name, mask,
                               codewords, erasure_cws, rs_info, result)
            return True, result
        return False, result
    except:
        return False, None


def decode_qr_multi(image_path, max_codes=3):
    """Decode up to max_codes QR codes from image.

    Returns: list of decoded strings (up to max_codes)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load {image_path}")

    patterns = find_finder_patterns(image)
    if len(patterns) < 3:
        raise ValueError(f"Found {len(patterns)} patterns, need at least 3")

    results = []
    used_patterns = set()  # Track which patterns have been used

    def patterns_overlap(combo, used):
        """Check if any pattern in combo was already used."""
        return any(i in used for i in combo)

    def mark_used(combo):
        """Mark patterns as used."""
        for i in combo:
            used_patterns.add(i)

    # Try all combinations, collect unique results
    if len(patterns) >= 3:
        for combo in combinations(range(len(patterns)), 3):
            if len(results) >= max_codes:
                break
            if patterns_overlap(combo, used_patterns):
                continue

            p1, p2, p3 = patterns[combo[0]], patterns[combo[1]], patterns[combo[2]]
            try:
                tl, tr, bl = identify_corners([p1, p2, p3])
                success, result = try_decode_patterns(image, tl, tr, bl)
                if success and result not in results:
                    results.append(result)
                    mark_used(combo)
                    print(f"Decoded QR {len(results)} using patterns {combo}")
            except:
                continue

    if not results:
        raise ValueError("No QR codes could be decoded")

    return results


def decode_qr(image_path):
    """Decode QR code from image. Returns first successful decode."""
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None: raise ValueError(f"Cannot load {image_path}")

    print("Finding finder patterns...")
    patterns = find_finder_patterns(image)
    if len(patterns) < 3: raise ValueError(f"Found {len(patterns)} patterns, need at least 3")

    # Try valid geometry groups first
    groups = group_finder_patterns(patterns)

    # If we have more than 3 patterns, try all combinations
    if len(patterns) > 3:
        for combo in combinations(range(len(patterns)), 3):
            p1, p2, p3 = patterns[combo[0]], patterns[combo[1]], patterns[combo[2]]
            try:
                tl, tr, bl = identify_corners([p1, p2, p3])
                success, result = try_decode_patterns(image, tl, tr, bl, patterns=patterns)
                if success:
                    print(f"Decoded using patterns {combo}")
                    return result
            except:
                continue

    # Try valid groups
    for tl, tr, bl in groups:
        success, result = try_decode_patterns(image, tl, tr, bl, patterns=patterns)
        if success:
            return result

    # Fallback: use first 3 patterns
    if len(patterns) >= 3:
        tl, tr, bl = identify_corners(patterns[:3])

    # Estimate version
    dist = np.sqrt((tl['center'][0]-tr['center'][0])**2 + (tl['center'][1]-tr['center'][1])**2)
    version = max(1, min(40, round(((dist / (np.sqrt(tl['area']) / 7) + 7) - 17) / 4)))

    # Find best version (use quick mode for speed)
    print(f"Detecting version (est: {version})...")
    best_v, best_s = version, 0
    for v in range(max(1, version-2), min(40, version+3)):
        corners = get_qr_corners(tl, tr, bl, version=v, image=image)
        m, _, _ = sample_matrix(image, corners, v, quick=True)
        size = v * 4 + 17
        # Score: finder patterns + timing
        FINDER = np.array([[1,1,1,1,1,1,1],[1,0,0,0,0,0,1],[1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,1,1,1,0,1],[1,0,0,0,0,0,1],[1,1,1,1,1,1,1]])
        s = np.sum(m[0:7,0:7]==FINDER) + np.sum(m[0:7,size-7:size]==FINDER) + np.sum(m[size-7:size,0:7]==FINDER)
        # Timing pattern
        for i in range(8, size-8):
            if m[6, i] == (1 if i % 2 == 0 else 0): s += 1
            if m[i, 6] == (1 if i % 2 == 0 else 0): s += 1
        if s > best_s: best_v, best_s = v, s
    version = best_v
    print(f"Detected version {version}")

    print("Sampling matrix...")
    # Use alignment pattern (if present) to accurately find BR corner
    corners = get_qr_corners(tl, tr, bl, version=version, image=image)
    matrix, erasure_mask, debug_imgs = sample_matrix(image, corners, version)

    print("Decoding...")
    ec_level, mask = read_format_info(matrix)
    EC_NAMES = {0: 'M', 1: 'L', 2: 'H', 3: 'Q'}
    ec_name = EC_NAMES.get(ec_level, '?')
    print(f"  Version: {version}, RS level: {ec_name} (mask {mask})")
    unmasked = unmask(matrix, mask)
    codewords = read_codewords(unmasked)
    size = version * 4 + 17
    erasure_cws = get_codeword_erasures(size, erasure_mask)

    result, rs_info = decode_with_rs(codewords, version, ec_level, erasure_codewords=erasure_cws)
    if debug_imgs:
        save_debug_all(image, patterns, corners,
                       debug_imgs[0], debug_imgs[1], debug_imgs[2],
                       matrix, erasure_mask, unmasked, version, ec_name, mask,
                       codewords, erasure_cws, rs_info, result)
    return result


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    path = args[0] if args else "/Users/zhangmiaochang/Desktop/IMG_3765.jpeg"

    if '--debug' in flags:
        base = os.path.splitext(os.path.basename(path))[0]
        DEBUG_DIR = os.path.join(os.path.dirname(path) or '.', f"{base}_debug")
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"Debug output -> {DEBUG_DIR}/")

    try:
        print(decode_qr(path))
    except Exception as e:
        print(f"Error: {e}")
