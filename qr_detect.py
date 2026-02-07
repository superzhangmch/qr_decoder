#!/usr/bin/env python3.11
"""
QR Code Decoder - Pure Python
Usage: python3.11 qr_decode.py <image_path>
"""

import cv2
import numpy as np
import os
from itertools import combinations

# ============================================================================
# QR DETECTION
# ============================================================================

def find_finder_patterns(image):
    """
    查找右下角之外的那三个“回”字定位块
    
    Find finder patterns with their corner points.

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

        # 注意: hierarchy.shape = (N, 4),  4 = [next, prev, first_child, parent], hierarchy 衡量来轮廓的包含关系. 所以下面几句能得到 square 的嵌套层数
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

    # Group concentric squares (same center, different sizes). 聚合同心的方框轮廓线
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
    """
    从回字块中, 把每个合法 qr code 的三个, 聚一起
    
    Group finder patterns that belong to the same QR code.

    Returns list of (tl, tr, bl) tuples for each detected QR code.
    """
    if len(patterns) < 3:
        return []

    def dist(p1, p2):
        return np.sqrt((p1['center'][0]-p2['center'][0])**2 +
                       (p1['center'][1]-p2['center'][1])**2)

    def is_valid_qr_geometry(p1, p2, p3):
        """
        Check if 3 patterns form valid QR geometry (right angle at TL).
        """
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

            if 60 < angle < 120 and ratio < 3: # 提取回字的中心构成的角, 角度 60~120; 两条边差距不过 3 倍
                return True
        return False

    # Try all combinations of 3 patterns
    valid_groups = []
    for combo in combinations(range(len(patterns)), 3): # 对识别出的所有回字块, 排列组合出所有三元组, 看是否是一个 qr-code
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


