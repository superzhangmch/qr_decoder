#!/usr/bin/env python3.11
"""
QR Code Decoder - Pure Python
Usage: python3.11 qr_decode.py <image_path>
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
# Global debug output directory (None = disabled)
DEBUG_DIR = None

from reed_solomon import ReedSolomon
from qr_detect import find_finder_patterns, identify_corners, group_finder_patterns, get_qr_corners

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

    # ==== 做网格对齐

    def score(m):
        s = np.sum(m[0:7, 0:7] == FINDER) + np.sum(m[0:7, size-7:size] == FINDER) + np.sum(m[size-7:size, 0:7] == FINDER)
        for i in range(8, size-8): # 检查 timing pattern 是否黑白交替，判断采样是否对齐
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
        print ("quick found")
        return best, None, None

    print ("fine search")

    # Fine search
    ox, oy, ms = best_p
    for dox in np.arange(-0.5, 0.6, 0.2):
        for doy in np.arange(-0.5, 0.6, 0.2):
            for dms in np.arange(-0.15, 0.2, 0.05):
                m = sample(ox+dox, oy+doy, ms+dms)
                s = score(m)
                if s > best_s: best, best_s, best_p = m, s, (ox+dox, oy+doy, ms+dms)

    # ====

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

    # Debug: draw grid lines and sample points on warped image
    warped_grid = None
    if DEBUG_DIR and not quick and warped_color is not None:
        ox, oy, ms = best_p
        warped_grid = warped_color.copy()
        # Draw vertical grid lines
        for i in range(size + 1):
            x = int(ox + i * ms)
            if 0 <= x < warp_size:
                cv2.line(warped_grid, (x, 0), (x, warp_size - 1), (0, 0, 255), 1)
        # Draw horizontal grid lines
        for i in range(size + 1):
            y = int(oy + i * ms)
            if 0 <= y < warp_size:
                cv2.line(warped_grid, (0, y), (warp_size - 1, y), (0, 0, 255), 1)
        # Draw sample points (center of each module)
        for r in range(size):
            y = int(oy + (r + 0.5) * ms)
            for c in range(size):
                x = int(ox + (c + 0.5) * ms)
                if 0 <= x < warp_size and 0 <= y < warp_size:
                    warped_grid[y, x] = (0, 0, 255)

    debug_imgs = (warped, binary, warped_color, warped_grid) if DEBUG_DIR and not quick else None
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
        module_size = np.sqrt(tl['area']) / 7  # 单个小方块(像素)的大小. 一个回形block是 7x7 的
        version0 = ((dist / module_size + 7) - 17) / 4
        version = max(1, min(40, round(version0)))
        print ("  version", version0, version, "module_size", module_size)

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
                from qr_debug import save_debug_all
                save_debug_all(DEBUG_DIR, image, patterns or [tl, tr, bl], corners,
                               debug_imgs[0], debug_imgs[1], debug_imgs[2], debug_imgs[3],
                               matrix, erasure_mask, unmasked, version, ec_name, mask,
                               codewords, erasure_cws, rs_info, result, is_data_module)
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

    # Try valid geometry groups
    groups = group_finder_patterns(patterns)
    for tl, tr, bl in groups:
        if len(results) >= max_codes:
            break
        combo = (patterns.index(tl), patterns.index(tr), patterns.index(bl))
        if patterns_overlap(combo, used_patterns):
            continue

        success, result = try_decode_patterns(image, tl, tr, bl)
        if success and result not in results:
            results.append(result)
            mark_used(combo)
            print(f"Decoded QR {len(results)} using patterns {combo}")

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

    # Try valid geometry groups
    groups = group_finder_patterns(patterns)
    try_cnt = 0
    for tl, tr, bl in groups:
        success, result = try_decode_patterns(image, tl, tr, bl, patterns=patterns)
        try_cnt += 1
        if success:
            print ("ok. try_cnt=", try_cnt)
            return result

    # ======

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
        from qr_debug import save_debug_all
        save_debug_all(DEBUG_DIR, image, patterns, corners,
                       debug_imgs[0], debug_imgs[1], debug_imgs[2], debug_imgs[3],
                       matrix, erasure_mask, unmasked, version, ec_name, mask,
                       codewords, erasure_cws, rs_info, result, is_data_module)
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
