import cv2 as cv
import numpy as np
import time


img = cv.imread('img2.png', cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not read 'img.png'")

H, W = img.shape
N_PIXELS = H * W

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)
blurred = cv.GaussianBlur(enhanced, (5, 5), 0)

print(f"Image: {W}x{H} = {N_PIXELS:,} pixels\n")

def nothing(x): pass

window = 'Structural Features (q to quit)'
cv.namedWindow(window)

# Corner params
cv.createTrackbar('maxCorners',   window, 500,  2000, nothing)
cv.createTrackbar('quality x1000',window, 5,    100,  nothing)
cv.createTrackbar('minDistance',  window, 15,   100,  nothing)
cv.createTrackbar('blockSize',    window, 11,   31,   nothing)

# Canny params
cv.createTrackbar('canny_low',    window, 50,   255,  nothing)
cv.createTrackbar('canny_high',   window, 150,  255,  nothing)

# Line detection params
cv.createTrackbar('line_thresh',   window, 50,  200,  nothing)
cv.createTrackbar('min_line_len',  window, 40,  300,  nothing)
cv.createTrackbar('max_line_gap',  window, 10,  100,  nothing)

# Line proximity
cv.createTrackbar('line_dist',    window, 10,  50,   nothing)

# Rectangle params
cv.createTrackbar('min_rect_area',window, 500, 10000, nothing)
cv.createTrackbar('rect_epsilon', window, 2,   20,   nothing)   # polygon approx tolerance

while True:
    max_corners  = max(1, cv.getTrackbarPos('maxCorners', window))
    quality      = max(cv.getTrackbarPos('quality x1000', window) / 1000.0, 0.001)
    min_distance = max(1, cv.getTrackbarPos('minDistance', window))
    block_size   = cv.getTrackbarPos('blockSize', window)
    block_size   = max(3, block_size if block_size % 2 == 1 else block_size + 1)

    canny_low    = cv.getTrackbarPos('canny_low', window)
    canny_high   = max(canny_low + 1, cv.getTrackbarPos('canny_high', window))

    line_thresh  = max(10, cv.getTrackbarPos('line_thresh', window))
    min_line_len = max(5, cv.getTrackbarPos('min_line_len', window))
    max_line_gap = cv.getTrackbarPos('max_line_gap', window)

    line_dist    = max(1, cv.getTrackbarPos('line_dist', window))

    min_rect_area = max(100, cv.getTrackbarPos('min_rect_area', window))
    rect_eps_pct  = max(1, cv.getTrackbarPos('rect_epsilon', window)) / 1000.0

    # ---- Edges ----
    t0 = time.perf_counter()
    edges = cv.Canny(blurred, canny_low, canny_high)
    t_canny = time.perf_counter() - t0

    # ---- Line detection (Hough) ----
    t0 = time.perf_counter()
    lines = cv.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=line_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    t_lines = time.perf_counter() - t0

    # ---- Line zone (dilate the lines) ----
    t0 = time.perf_counter()
    line_mask = np.zeros_like(edges)
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv.line(line_mask, (x1, y1), (x2, y2), 255, thickness=1)
    line_zone = cv.dilate(line_mask, np.ones((line_dist, line_dist), np.uint8))
    t_linezone = time.perf_counter() - t0

    # ---- Rectangle detection ----
    t0 = time.perf_counter()
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_rect_area:
            continue
        perim = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, rect_eps_pct * perim, True)
        if len(approx) == 4 and cv.isContourConvex(approx):
            rects.append(approx)
    t_rects = time.perf_counter() - t0

    # ---- Rectangle zone (filled polygons dilated) ----
    rect_mask = np.zeros_like(edges)
    for r in rects:
        cv.drawContours(rect_mask, [r], -1, 255, thickness=cv.FILLED)
    rect_zone = cv.dilate(rect_mask, np.ones((line_dist, line_dist), np.uint8))

    # Combined structural zone
    struct_zone = cv.bitwise_or(line_zone, rect_zone)

    # ---- Corner detection ----
    t0 = time.perf_counter()
    cand = cv.goodFeaturesToTrack(
        blurred,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
        mask=None
    )
    t_corners = time.perf_counter() - t0

    # ---- Filter corners by structural zone ----
    t0 = time.perf_counter()
    n_total = 0
    n_structural = 0
    if cand is not None:
        cand = cand.reshape(-1, 2)
        xs = np.clip(cand[:, 0].astype(int), 0, W - 1)
        ys = np.clip(cand[:, 1].astype(int), 0, H - 1)
        on_struct = struct_zone[ys, xs] > 0
        n_total = len(cand)
        n_structural = int(on_struct.sum())
    t_filter = time.perf_counter() - t0

    t_total = t_canny + t_lines + t_linezone + t_rects + t_corners + t_filter

    # ---- Visualization ----
    vis = cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)

    # Rectangle zone (yellow tint)
    yellow = np.array([0, 180, 180])
    mask3 = rect_zone > 0
    vis[mask3] = (0.5 * vis[mask3] + 0.5 * yellow).astype(np.uint8)

    # Line zone (cyan tint)
    cyan = np.array([180, 180, 0])
    mask3 = (line_zone > 0) & ~(rect_zone > 0)
    vis[mask3] = (0.5 * vis[mask3] + 0.5 * cyan).astype(np.uint8)

    # Actual edges (dim blue)
    vis[edges > 0] = (200, 100, 50)

    # Hough lines (bright cyan)
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Rectangles (bright yellow)
    for r in rects:
        cv.drawContours(vis, [r], -1, (0, 255, 255), 2)

    # Corners
    if cand is not None:
        for (x, y) in cand[~on_struct].astype(int):
            cv.circle(vis, (x, y), 3, (0, 0, 200), 1)   # red = rejected
        for (x, y) in cand[on_struct].astype(int):
            cv.circle(vis, (x, y), 4, (0, 255, 0), 2)   # green = kept

    # HUD
    kept_pct = (n_structural / n_total * 100) if n_total else 0
    n_lines = 0 if lines is None else len(lines)
    hud = [
        f"corners: {n_total}  structural: {n_structural} ({kept_pct:.0f}%)",
        f"lines: {n_lines}  rectangles: {len(rects)}",
        f"canny=({canny_low},{canny_high}) hough_thr={line_thresh} min_len={min_line_len}",
        f"corner: q={quality:.3f} block={block_size} minDist={min_distance}",
        "",
        f"TIMING:",
        f"  canny    : {t_canny*1000:6.2f} ms  O(N)",
        f"  hough    : {t_lines*1000:6.2f} ms  O(E * theta_bins)",
        f"  line zone: {t_linezone*1000:6.2f} ms  O(N) draw + O(N*k^2) dilate",
        f"  rects    : {t_rects*1000:6.2f} ms  O(E_contour) + O(V^2) approx",
        f"  corners  : {t_corners*1000:6.2f} ms  O(N*b^2)",
        f"  filter   : {t_filter*1000:6.2f} ms  O(C)",
        f"  TOTAL    : {t_total*1000:6.2f} ms  (FPS cap: {1.0/max(t_total, 1e-6):.1f})",
    ]
    for i, text in enumerate(hud):
        y = 20 + i * 18
        cv.putText(vis, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv.putText(vis, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv.imshow(window, vis)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Final report ----
print("\n" + "=" * 60)
print("FINAL PARAMETERS")
print("=" * 60)
print(f"Corners:  maxCorners={max_corners}, quality={quality:.4f},")
print(f"          minDistance={min_distance}, blockSize={block_size}")
print(f"Canny:    low={canny_low}, high={canny_high}")
print(f"Hough:    threshold={line_thresh}, minLength={min_line_len}, maxGap={max_line_gap}")
print(f"Rects:    min_area={min_rect_area}, epsilon_pct={rect_eps_pct}")
print(f"Zone:     dilate={line_dist}")

print("\n" + "=" * 60)
print("COMPUTATIONAL COMPLEXITY")
print("=" * 60)
print(f"N = {N_PIXELS:,} pixels, E = edge pixels, C = {n_total} corners")
print()
print(f"Canny:     O(N)                      ~ {N_PIXELS * 10:,} ops")
print(f"Hough:     O(E * theta_bins) + O(peaks)")
print(f"           E~{(edges>0).sum():,} edge pixels, 180 angle bins")
print(f"Contours:  O(E) traversal + O(V^2) polygon approx per contour")
print(f"Corners:   O(N * blockSize^2)        ~ {N_PIXELS * block_size**2:,} ops")
print(f"Filter:    O(C)                      ~ {n_total * 5} ops")
print(f"Total:     {t_total*1000:.2f} ms measured  → max {1.0/max(t_total, 1e-6):.1f} FPS")

cv.destroyAllWindows()