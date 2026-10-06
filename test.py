import cv2 as cv
import numpy as np


img = cv.imread('img.png')
if img is None:
    raise FileNotFoundError("Could not read 'img.png'")

channels = {
    'Gray':  cv.cvtColor(img, cv.COLOR_BGR2GRAY),
    'Blue':  img[:, :, 0],
    'Green': img[:, :, 1],
    'Red':   img[:, :, 2],
}


def nothing(x): pass


window = 'Tuning (q to quit)'
cv.namedWindow(window)

# --- Preprocessing ---
cv.createTrackbar('channel (0=G 1=B 2=Grn 3=R)', window, 0,  3,  nothing)
cv.createTrackbar('clahe clip x10',              window, 20, 100, nothing)   # 0.1 to 10
cv.createTrackbar('clahe tile',                  window, 8,  32, nothing)
cv.createTrackbar('blur size (odd, 0=off)',      window, 5,  21, nothing)
cv.createTrackbar('gamma x100',                  window, 100,300, nothing)   # 0.3 to 3.0

# --- Canny ---
cv.createTrackbar('canny_low',                   window, 50, 255, nothing)
cv.createTrackbar('canny_high',                  window, 150,255, nothing)
cv.createTrackbar('dilate kernel',               window, 5,  51, nothing)

# --- goodFeaturesToTrack ---
cv.createTrackbar('maxCorners',                  window, 500,2000,nothing)
cv.createTrackbar('quality x1000',               window, 5,  100, nothing)   # 0.001 to 0.1
cv.createTrackbar('minDistance',                 window, 15, 100, nothing)
cv.createTrackbar('blockSize (odd)',             window, 11, 31, nothing)

# --- Optical flow (shown as info; apply when you have 2 frames) ---
cv.createTrackbar('LK winSize (odd)',            window, 21, 61, nothing)
cv.createTrackbar('LK maxLevel',                 window, 3,  6,  nothing)
cv.createTrackbar('LK iterations',               window, 30, 100,nothing)
cv.createTrackbar('LK eps x1000',                window, 10, 100,nothing)   # 0.001 to 0.1

channel_names = list(channels.keys())
clahe_cache = {}


def get_clahe(clip, tile):
    key = (round(clip, 2), tile)
    if key not in clahe_cache:
        clahe_cache[key] = cv.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe_cache[key]


while True:
    # Read all parameters
    ch_idx       = cv.getTrackbarPos('channel (0=G 1=B 2=Grn 3=R)', window)
    clip         = max(0.1, cv.getTrackbarPos('clahe clip x10', window) / 10.0)
    tile         = max(2, cv.getTrackbarPos('clahe tile', window))
    blur_k       = cv.getTrackbarPos('blur size (odd, 0=off)', window)
    blur_k       = 0 if blur_k == 0 else (blur_k if blur_k % 2 == 1 else blur_k + 1)
    gamma        = max(0.1, cv.getTrackbarPos('gamma x100', window) / 100.0)

    canny_lo     = cv.getTrackbarPos('canny_low', window)
    canny_hi     = max(canny_lo + 1, cv.getTrackbarPos('canny_high', window))
    dil_k        = cv.getTrackbarPos('dilate kernel', window)
    dil_k        = max(1, dil_k if dil_k % 2 == 1 else dil_k + 1)

    max_corners  = max(1, cv.getTrackbarPos('maxCorners', window))
    quality      = max(0.001, cv.getTrackbarPos('quality x1000', window) / 1000.0)
    min_dist     = max(1, cv.getTrackbarPos('minDistance', window))
    block_sz     = cv.getTrackbarPos('blockSize (odd)', window)
    block_sz     = max(3, block_sz if block_sz % 2 == 1 else block_sz + 1)

    lk_win       = cv.getTrackbarPos('LK winSize (odd)', window)
    lk_win       = max(5, lk_win if lk_win % 2 == 1 else lk_win + 1)
    lk_level     = cv.getTrackbarPos('LK maxLevel', window)
    lk_iter      = max(1, cv.getTrackbarPos('LK iterations', window))
    lk_eps       = max(0.001, cv.getTrackbarPos('LK eps x1000', window) / 1000.0)

    # Pick source channel
    name = channel_names[ch_idx]
    src = channels[name]

    # --- Preprocess pipeline ---
    clahe = get_clahe(clip, tile)
    processed = clahe.apply(src)

    # Gamma
    if abs(gamma - 1.0) > 0.01:
        table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
        processed = cv.LUT(processed, table)

    # Blur
    if blur_k > 0:
        processed = cv.GaussianBlur(processed, (blur_k, blur_k), 0)

    # --- Canny + dilate ---
    edges = cv.Canny(processed, canny_lo, canny_hi)
    edge_zone = cv.dilate(edges, np.ones((dil_k, dil_k), np.uint8))

    # --- Corner detection ---
    cand = cv.goodFeaturesToTrack(
        processed, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_dist, blockSize=block_sz, mask=None
    )

    # --- Visualization ---
    vis = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)

    # Dim blue tint for edge zone, bright blue for actual edges
    tint = np.array([150, 50, 0])
    mask3 = edge_zone > 0
    vis[mask3] = (0.5 * vis[mask3] + 0.5 * tint).astype(np.uint8)
    vis[edges > 0] = (255, 100, 0)

    n_total = n_on_edge = 0
    if cand is not None:
        cand = cand.reshape(-1, 2)
        xs = np.clip(cand[:, 0].astype(int), 0, vis.shape[1] - 1)
        ys = np.clip(cand[:, 1].astype(int), 0, vis.shape[0] - 1)
        near = edge_zone[ys, xs] > 0
        n_total = len(cand)
        n_on_edge = int(near.sum())

        for (x, y) in cand[~near].astype(int):
            cv.circle(vis, (x, y), 3, (0, 0, 200), 1)
        for (x, y) in cand[near].astype(int):
            cv.circle(vis, (x, y), 4, (0, 255, 0), 2)

    # HUD
    edge_frac = (edges > 0).mean()
    kept = (n_on_edge / n_total * 100) if n_total else 0
    hud = [
        f"channel: {name}",
        f"corners: {n_total}  on-edge: {n_on_edge} ({kept:.0f}%)  edges: {edge_frac:.1%}",
        f"CLAHE clip={clip:.1f} tile={tile}  gamma={gamma:.2f}  blur={blur_k}",
        f"Canny=({canny_lo},{canny_hi}) dilate={dil_k}",
        f"GFTT q={quality:.3f} min_d={min_dist} block={block_sz} max={max_corners}",
        f"LK win={lk_win} lvl={lk_level} iter={lk_iter} eps={lk_eps:.3f}",
    ]
    for i, text in enumerate(hud):
        y = 20 + i * 20
        cv.putText(vis, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv.putText(vis, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv.imshow(window, vis)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Final report
print("\n--- FINAL PARAMETERS ---")
print(f"Channel        = {name}")
print(f"CLAHE          = clipLimit={clip}, tileGridSize=({tile},{tile})")
print(f"Blur           = {blur_k}")
print(f"Gamma          = {gamma}")
print(f"Canny          = ({canny_lo}, {canny_hi})")
print(f"Dilate kernel  = {dil_k}")
print(f"maxCorners     = {max_corners}")
print(f"qualityLevel   = {quality}")
print(f"minDistance    = {min_dist}")
print(f"blockSize      = {block_sz}")
print(f"LK winSize     = ({lk_win}, {lk_win})")
print(f"LK maxLevel    = {lk_level}")
print(f"LK iterations  = {lk_iter}")
print(f"LK eps         = {lk_eps}")

cv.destroyAllWindows()