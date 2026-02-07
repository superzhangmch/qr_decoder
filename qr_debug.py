"""QR decode debug visualization - saves intermediate results to disk."""

import cv2
import numpy as np
import os


def _save_img(debug_dir, name, data, scale=None):
    """Save image to debug_dir."""
    path = os.path.join(debug_dir, name)
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
    for c in range(9): t[8, c] = 5
    for c in range(size-8, size): t[8, c] = 5
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


def _draw_colored_matrix(matrix, version, is_data_module):
    """Draw QR matrix with different colors for each functional region."""
    size = matrix.shape[0]
    scale = 20
    tmap = _module_type_map(version)

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
                val = 255 if matrix[r, c] == 0 else 0
                vis[y0:y1, x0:x1] = val
            else:
                color = COLORS.get(mt, (128, 128, 128))
                if matrix[r, c] == 1:
                    vis[y0:y1, x0:x1] = color
                else:
                    vis[y0:y1, x0:x1] = tuple(min(255, int(v * 0.4 + 255 * 0.6)) for v in color)
            vis[y0, x0:x1] = (60, 60, 60)
            vis[y0:y1, x0] = (60, 60, 60)

    # Draw actual zigzag traversal path through data modules
    half = scale // 2
    path = []
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

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        x1, y1 = c1 * scale + half, r1 * scale + half
        x2, y2 = c2 * scale + half, r2 * scale + half
        t = i / max(len(path) - 1, 1)
        color = (0, int(200 * (1 - t)), int(200 * t))
        cv2.line(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    if path:
        sr, sc = path[0]
        cv2.circle(vis, (sc * scale + half, sr * scale + half), 4, (0, 255, 0), -1)
        er, ec = path[-1]
        cv2.circle(vis, (ec * scale + half, er * scale + half), 4, (0, 0, 255), -1)

    return vis


def save_debug_all(debug_dir, image, patterns, corners, warped_gray, warped_binary, warped_color, warped_grid,
                   matrix, erasure_mask, unmasked, version, ec_name, mask, codewords,
                   erasure_cws, rs_info, result, is_data_module):
    """Save all intermediate results to debug_dir."""
    if not debug_dir:
        return

    # 1: detected finder patterns + QR boundary
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, p in enumerate(patterns):
        cx, cy = int(p['center'][0]), int(p['center'][1])
        cv2.drawContours(vis, [p['contour']], -1, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (cx+8, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if p.get('corners') is not None:
            for px, py in p['corners']:
                cv2.circle(vis, (int(px), int(py)), 4, (255, 0, 0), -1)
    pts = corners.astype(int)
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 255), 2)
    labels = ['TL', 'TR', 'BR', 'BL']
    for i in range(4):
        cv2.circle(vis, tuple(pts[i]), 8, (0, 255, 255), -1)
        cv2.putText(vis, labels[i], (pts[i][0]+10, pts[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    _save_img(debug_dir, "1_detected.png", vis)

    # 2: warped (gray + binary + color side-by-side)
    gray3 = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    bin3 = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
    panels = [gray3, bin3] + ([warped_color] if warped_color is not None else [])
    _save_img(debug_dir, "2_warped.png", np.hstack(panels))

    # 2b: warped with grid lines
    if warped_grid is not None:
        _save_img(debug_dir, "2b_grid.png", warped_grid)

    # 3: sampled matrix with color-coded function regions
    _save_img(debug_dir, "3_matrix.png", _draw_colored_matrix(matrix, version, is_data_module))

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
        _save_img(debug_dir, f"{n}_erasures.png", vis); n += 1

    # unmasked matrix
    _save_img(debug_dir, f"{n}_unmasked.png", unmasked); n += 1

    # RS correction details
    rs_lines = []
    for b in rs_info:
        rs_lines.append(f"  Block {b['block']}: {b['data_len']}+{b['ec_len']} bytes, "
                        f"{b['errors']} errors, {b['erasures']} erasures -> {b['status']}")
    rs_text = '\n'.join(rs_lines)
    with open(os.path.join(debug_dir, f"{n}_rs.txt"), 'w') as f:
        f.write(rs_text + '\n')
    n += 1

    # info + result
    size = version * 4 + 17
    info = (f"Version: {version}\nEC level: {ec_name}\nMask: {mask}\n"
            f"Size: {size}x{size}\nCodewords: {len(codewords)}\n"
            f"Erasure codewords: {len(erasure_cws) if erasure_cws else 0}\n"
            f"\nRS correction:\n{rs_text}\n"
            f"\nResult:\n{result}\n")
    with open(os.path.join(debug_dir, f"{n}_info.txt"), 'w') as f:
        f.write(info)
