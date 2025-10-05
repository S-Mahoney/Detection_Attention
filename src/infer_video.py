# media_grid_rfdetr.py
# ------------------------------------------------------------
# Supports: single image, directory of images (as frames), or MP4 video.
# Outputs:  - For single image: composite PNG.
#           - For dir/video: composite MP4 (and optional per-frame PNGs via --save_frames).
#
# Examples:
#   Single image  -> PNG:
#     python media_grid_rfdetr.py --source path/to/img.jpg --target out/grid.png
#
#   Directory of images -> MP4:
#     python media_grid_rfdetr.py --source path/to/frames_dir --target out/grid.mp4 --cols 3 --segment --classify
#
#   MP4 video -> MP4:
#     python media_grid_rfdetr.py --source in/video.mp4 --target out/grid.mp4 --cols 3 --segment
# ------------------------------------------------------------
import argparse, math, os, glob, colorsys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import supervision as sv

from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES


# -------------------- feature tapper (hooks) --------------------
class FeatureTapper:
    """Hook 4D (B,C,H,W) tensors; used to grab 'final' feature maps per frame."""
    def __init__(self, torch_module, patterns=("input_proj","backbone","neck","fpn","out","encoder")):
        self.m = torch_module
        self.patterns = tuple(s.lower() for s in patterns)
        self.handles = []
        self.captured = {}

    def _should(self, name: str):
        nl = name.lower()
        return any(p in nl for p in self.patterns)

    def _store_any(self, name, out):
        if torch.is_tensor(out) and out.dim() == 4:
            self.captured[name] = out.detach().cpu(); return
        tens = getattr(out, "tensors", None)
        if torch.is_tensor(tens) and tens.dim() == 4:
            self.captured[name] = tens.detach().cpu(); return
        if isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                if torch.is_tensor(o) and o.dim() == 4:
                    self.captured[f"{name}_{i}"] = o.detach().cpu()
                else:
                    t2 = getattr(o, "tensors", None)
                    if torch.is_tensor(t2) and t2.dim() == 4:
                        self.captured[f"{name}_{i}"] = t2.detach().cpu()

    def _hook(self, name):
        def fn(_m, _inp, out):
            try: self._store_any(name, out)
            except Exception: pass
        return fn

    def attach(self):
        for name, module in self.m.named_modules():
            if self._should(name):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def detach(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()

    def clear(self):
        self.captured.clear()


def pick_final_keys(captured: dict[str, torch.Tensor]) -> list[str]:
    keys = list(captured.keys())
    p1 = [k for k in keys if "input_proj" in k.lower()]
    if p1: return sorted(p1)
    p2 = [k for k in keys if k.startswith("backbone.0.encoder_")]
    if p2: return sorted(p2)  # usually 4 levels
    p3 = [k for k in keys if k.startswith("backbone.0.projector_0")]
    if p3: return sorted(p3)
    p4 = [k for k in keys if k.startswith("backbone.1")]
    return sorted(p4)


# -------------------- fixed PCA per level (stable colors) --------------------
class PCAMapper:
    """Fit PCA (via SVD) on first frame; apply same basis to all frames (stable colors)."""
    def __init__(self):
        self.mean = None  # (C,)
        self.V = None     # (C,3)

    def fit(self, feat_bchw: torch.Tensor):
        if feat_bchw.dim() == 4: feat = feat_bchw[0]
        else: feat = feat_bchw
        C, H, W = feat.shape
        X = feat.reshape(C, H*W).t()
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        self.mean = mu.squeeze(0).cpu()
        self.V = Vt.t()[:, :3].cpu()

    def transform(self, feat_bchw: torch.Tensor) -> np.ndarray:
        if self.V is None: raise RuntimeError("PCAMapper not fitted")
        if feat_bchw.dim() == 4: feat = feat_bchw[0]
        else: feat = feat_bchw
        C, H, W = feat.shape
        X = feat.reshape(C, H*W).t().cpu()
        Xc = X - self.mean
        comp = Xc @ self.V
        comp = comp.reshape(H, W, 3)
        mn = comp.amin(axis=(0,1), keepdims=True)
        mx = comp.amax(axis=(0,1), keepdims=True)
        comp = (comp - mn) / (mx - mn + 1e-6)
        return comp.numpy()  # RGB [0,1]


# -------------------- segmentation / grouping --------------------
def _kmeans_np(X, k=6, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.choice(N, size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        L = d2.argmin(1)
        newC = []
        for j in range(k):
            m = X[L == j]
            if len(m) == 0: newC.append(X[rng.integers(0, N)])
            else: newC.append(m.mean(0))
        newC = np.stack(newC, 0)
        if np.allclose(newC, C): C = newC; break
        C = newC
    return L, C

def _label_edges(labels):
    H, W = labels.shape
    e = np.zeros((H, W), bool)
    e[:-1, :] |= labels[:-1, :] != labels[1:, :]
    e[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return e

def stack_resized_pca_rgbs(pca_rgbs: dict, W: int, H: int) -> np.ndarray:
    ups = []
    for name in sorted(pca_rgbs.keys()):
        arr = np.clip(pca_rgbs[name], 0, 1)
        im = Image.fromarray((arr*255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)
        ups.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.concatenate(ups, axis=2)

def unified_segmentation_from_pca(pca_rgbs: dict, img_size: tuple[int,int], k=6, seed=0):
    W, H = img_size
    feat = stack_resized_pca_rgbs(pca_rgbs, W, H).astype(np.float32)  # HxWx(3*L)
    X = feat.reshape(-1, feat.shape[2])
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    labels_1d, _ = _kmeans_np(X, k=k, iters=25, seed=seed)
    return labels_1d.reshape(H, W).astype(np.int32)

def group_segments_into_classes(labels: np.ndarray, pca_rgbs: dict, img_size: tuple[int,int],
                                method="agglom", class_k=None, dist_thresh=0.15, min_seg_px=64):
    # Build z-scored features
    W, H = img_size
    feat = stack_resized_pca_rgbs(pca_rgbs, W, H).astype(np.float32)  # HxWxC
    Hh, Wh, C = feat.shape
    X = feat.reshape(-1, C)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    feat_z = X.reshape(Hh, Wh, C)

    seg_ids = np.arange(labels.max() + 1)
    seg_vecs, areas = [], []
    for s in seg_ids:
        m = (labels == s)
        a = int(m.sum()); areas.append(a)
        if a < min_seg_px: seg_vecs.append(None); continue
        seg_vecs.append(feat_z[m].mean(0))
    keep = [i for i,v in enumerate(seg_vecs) if v is not None]
    if not keep:
        return None, None, None
    seg_vecs = np.stack([seg_vecs[i] for i in keep], 0)

    if method == "kmeans" and class_k:
        L, _ = _kmeans_np(seg_vecs, k=class_k, iters=50, seed=0)
        seg2cls = {seg_ids[keep[i]]: int(L[i]) for i in range(len(keep))}
        clusters = []
        for c in range(class_k):
            members = [seg_ids[keep[i]] for i in range(len(keep)) if L[i] == c]
            if members: clusters.append((None, members))
    else:
        # greedy agglomerative cosine
        clusters = [(seg_vecs[i].copy(), [i]) for i in range(seg_vecs.shape[0])]
        while True:
            n = len(clusters)
            if n <= 1: break
            V = np.stack([c[0] for c in clusters], 0)
            V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
            dist = 1.0 - (V @ V.T)
            np.fill_diagonal(dist, np.inf)
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            if not np.isfinite(dist[i, j]) or dist[i, j] >= dist_thresh: break
            merged_members = clusters[i][1] + clusters[j][1]
            merged_vec = seg_vecs[np.array(merged_members)].mean(0)
            keep_idx = [idx for idx in range(n) if idx not in (i, j)]
            clusters = [clusters[idx] for idx in keep_idx] + [(merged_vec, merged_members)]
        mapping = {}
        for cid, (_, members_idx) in enumerate(clusters):
            for m in members_idx: mapping[m] = cid
        seg2cls = {seg_ids[keep[i]]: int(mapping[i]) for i in range(len(keep))}

    # Build class_map from seg2cls
    class_map = np.full_like(labels, 255, dtype=np.int32)
    for s, cid in seg2cls.items():
        class_map[labels == s] = cid

    return class_map, seg2cls, clusters

def _stack_pca_to_common_grid(pca_rgbs: dict[str, np.ndarray],
                              mode: str = "max") -> tuple[np.ndarray, tuple[int, int]]:
    """
    pca_rgbs: {name: HxWx3 float32 in [0,1]}
    mode='max'  → upsample every level to the largest (H,W) among levels
    mode='min'  → downsample to the smallest (H,W)
    Returns: (feat_hwC, (Wc,Hc)) where C = 3 * num_levels
    """
    assert pca_rgbs, "no PCA maps"
    sizes = [(arr.shape[1], arr.shape[0]) for arr in pca_rgbs.values()]  # (W,H)
    if mode == "max":
        Wc = max(w for w, h in sizes); Hc = max(h for w, h in sizes)
    else:
        Wc = min(w for w, h in sizes); Hc = min(h for w, h in sizes)

    ups = []
    for name in sorted(pca_rgbs.keys()):
        arr = np.clip(pca_rgbs[name], 0, 1)
        im = Image.fromarray((arr * 255).astype(np.uint8)).resize((Wc, Hc), resample=Image.BILINEAR)
        ups.append(np.asarray(im, dtype=np.float32) / 255.0)   # Hc x Wc x 3
    feat = np.concatenate(ups, axis=2).astype(np.float32)      # Hc x Wc x (3*L)
    return feat, (Wc, Hc)

def unified_segmentation_on_feature_grid(pca_rgbs: dict[str, np.ndarray],
                                         k: int = 6,
                                         seed: int = 0,
                                         mode: str = "max") -> tuple[np.ndarray, tuple[int,int]]:
    """
    Cluster in *feature space* (common feature grid), not image space.
    Returns: (labels_feat Hc x Wc, (Wc,Hc))
    """
    feat, (Wc, Hc) = _stack_pca_to_common_grid(pca_rgbs, mode=mode)
    X = feat.reshape(-1, feat.shape[2])
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    labels_1d, _ = _kmeans_np(X, k=k, iters=25, seed=seed)
    labels = labels_1d.reshape(Hc, Wc).astype(np.int32)
    return labels, (Wc, Hc)

def colorize_ids(labels: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    """
    labels: HxW ints
    palette: Kx3 uint8, if None we generate a stable palette
    """
    K = int(labels.max()) + 1
    if palette is None:
        rng = np.random.default_rng(0)
        palette = rng.integers(0, 255, size=(K, 3), dtype=np.uint8)
        palette[0] = np.array([30, 30, 30], np.uint8)  # make id 0 darker
    palette = palette[:K]
    return palette[labels]

def _calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    X: (N,D) float32  | labels: (N,) ints in [0..k-1]
    """
    n, d = X.shape
    labs = labels.astype(np.int32)
    k = int(labs.max()) + 1
    if k < 2 or k >= n: 
        return 0.0
    overall = X.mean(axis=0, keepdims=True)
    B = 0.0
    W = 0.0
    for j in range(k):
        mask = (labs == j)
        if not mask.any(): 
            continue
        Xj = X[mask]
        cj = Xj.mean(axis=0, keepdims=True)
        B += len(Xj) * float(np.sum((cj - overall) ** 2))
        W += float(np.sum((Xj - cj) ** 2))
    # (B/(k-1)) / (W/(n-k))
    return (B / max(k - 1, 1)) / (W / max(n - k, 1) + 1e-12 + 0.0)

def _feature_grid_from_pca_rgbs(pca_rgbs: dict[str, np.ndarray], mode: str = "max"):
    """
    Stack the 4 PCA→RGB maps onto a common feature grid (HxWx12), z-score per channel,
    and also return per-level grayscale gradients (for a separation prior).
    """
    feat, (Wc, Hc) = _stack_pca_to_common_grid(pca_rgbs, mode=mode)  # HxWx(3*L), [0,1]
    # z-score channels
    X = feat.reshape(-1, feat.shape[2]).astype(np.float32)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    feat_z = X.reshape(feat.shape)

    # separation prior via average Sobel gradient magnitude across the 4 maps
    import cv2
    grads = []
    for name in sorted(pca_rgbs.keys()):
        arr = np.clip(pca_rgbs[name], 0, 1)
        im = cv2.resize(arr, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        # normalize inside-frame to [0,1] to be scale-agnostic
        grads.append((g.mean() / (g.max() + 1e-8)))
    sep_score = float(np.mean(grads))  # 0..~1

    return feat_z, (Wc, Hc), sep_score

def estimate_k_from_color_separation(
        pca_rgbs: dict[str, np.ndarray],
        kmin: int = 3, kmax: int = 15,
        sample_px: int = 30000,  # speed
        iters: int = 20, seed: int = 0,
        mode: str = "max"):
    """
    1) Compute an average separation prior from gradients across levels.
    2) Use it to bound the candidate range.
    3) Pick k by maximizing Calinski–Harabasz on a sampled set (k-means via your _kmeans_np).
    Returns: k_best, debug_info
    """
    rng = np.random.default_rng(seed)
    feat_z, (Wc, Hc), sep = _feature_grid_from_pca_rgbs(pca_rgbs, mode=mode)  # HxWxC, sep∈[0,1]

    # candidate band from separation prior (more separation → allow bigger k)
    k_lo = max(2, int(round(kmin)))
    k_hi = int(round(kmin + sep * (kmax - kmin)))
    k_hi = max(k_hi, k_lo + 1)
    k_hi = min(k_hi, kmax)

    # sample pixels for speed
    N = feat_z.shape[0] * feat_z.shape[1]
    idx = rng.choice(N, size=min(sample_px, N), replace=False)
    Xs = feat_z.reshape(N, -1)[idx].astype(np.float32)

    scores = []
    tried = []
    for k in range(k_lo, k_hi + 1):
        if k < 2: 
            continue
        labs, _ = _kmeans_np(Xs, k=k, iters=iters, seed=int(seed + k))
        score = _calinski_harabasz_score(Xs, labs)
        scores.append(score); tried.append(k)

    if not scores:
        return max(2, kmin), {"sep": sep, "band": (k_lo, k_hi), "scores": []}

    k_best = tried[int(np.argmax(scores))]
    return k_best, {"sep": sep, "band": (k_lo, k_hi), "scores": list(zip(tried, scores))}


# -------------------- detection overlap → COCO class coloring --------------------
def coco_color_palette(n=91, sat=0.65, val=1.0):
    # deterministic HSV palette
    cols = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        cols.append((int(r*255), int(g*255), int(b*255)))  # RGB
    return np.array(cols, dtype=np.uint8)

COCO_COLORS = coco_color_palette(91)

def map_regions_to_detection_classes(region_map: np.ndarray, dets: sv.Detections, img_wh: tuple[int,int]) -> dict:
    """Return {region_id -> coco_class_id or -1} using pixel vote overlap with detection boxes."""
    W, H = img_wh
    assign = {}
    # Pre-bin detections by class
    by_class = {}
    for i, cid in enumerate(dets.class_id):
        by_class.setdefault(int(cid), []).append(dets.xyxy[i])
    # For each region id, vote pixels from all boxes per class
    region_ids = np.unique(region_map[region_map >= 0])
    for rid in region_ids:
        mask = (region_map == rid)
        votes = {}
        for cid, boxes in by_class.items():
            cnt = 0
            for (x1, y1, x2, y2) in boxes:
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(W, int(x2)), min(H, int(y2))
                if xi2 > xi1 and yi2 > yi1:
                    sub = mask[yi1:yi2, xi1:xi2]
                    cnt += int(sub.sum())
            if cnt > 0: votes[cid] = votes.get(cid, 0) + cnt
        if votes:
            best = max(votes.items(), key=lambda kv: kv[1])[0]
            assign[int(rid)] = int(best)
        else:
            assign[int(rid)] = -1
    return assign

def color_map_from_class_ids(region_map: np.ndarray, rid2cls: dict, default_rgb=(64,64,64)) -> np.ndarray:
    H, W = region_map.shape
    out = np.zeros((H, W, 3), np.uint8)
    out[:] = np.array(default_rgb, np.uint8)
    for rid, cls in rid2cls.items():
        if cls >= 0 and cls < len(COCO_COLORS):
            out[region_map == rid] = COCO_COLORS[cls]
    return out


# -------------------- drawing / tiling --------------------
def scale_detections(dets: sv.Detections, src_wh: tuple[int,int], dst_wh: tuple[int,int]) -> sv.Detections:
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    xyxy = dets.xyxy.copy().astype(np.float32)
    xyxy[:, [0, 2]] *= sx
    xyxy[:, [1, 3]] *= sy
    return sv.Detections(xyxy=xyxy, confidence=dets.confidence.copy(), class_id=dets.class_id.copy())

def to_bgr_uint8(rgb01: np.ndarray) -> np.ndarray:
    arr = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    return arr[:, :, ::-1]

def tile_grid(frames_bgr: list[np.ndarray], tiles_per_row: int | None) -> np.ndarray:
    assert len(frames_bgr) > 0
    H, W = frames_bgr[0].shape[:2]
    for f in frames_bgr:
        if f.shape[:2] != (H, W):
            raise ValueError("All tiles must share the same HxW")
    if tiles_per_row is None: tiles_per_row = len(frames_bgr)
    rows = int(math.ceil(len(frames_bgr) / tiles_per_row))
    cols = tiles_per_row
    pads = rows*cols - len(frames_bgr)
    if pads > 0: frames_bgr += [np.zeros_like(frames_bgr[0])] * pads
    rows_bgr = []
    for r in range(rows):
        rows_bgr.append(np.hstack(frames_bgr[r*cols:(r+1)*cols]))
    return np.vstack(rows_bgr)

def ensure_bgr8(img: np.ndarray) -> np.ndarray:
    """
    Ensure img is C-contiguous, writeable, uint8, 3-channel BGR (H,W,3).
    """
    if img is None:
        raise ValueError("ensure_bgr8 got None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # drop alpha
        img = img[:, :, :3]
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)

    # force contiguous + writeable
    if not img.flags.c_contiguous or not img.flags.writeable:
        img = np.ascontiguousarray(img)
        img.setflags(write=True)
    return img


# -------------------- IO helpers --------------------
def is_video(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def is_image(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

def list_images_in_dir(d: str) -> list[str]:
    exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"]
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(d, e)))
    return sorted(files)

def make_even(x: int) -> int:
    return x if (x % 2 == 0) else (x - 1)

def fit_size(W: int, H: int, max_w: int, max_h: int) -> tuple[int,int,float]:
    """Return (W2, H2, scale) where dims are <= max and even."""
    s = min(1.0, max_w / max(1, W), max_h / max(1, H))
    W2 = make_even(int(round(W * s)))
    H2 = make_even(int(round(H * s)))
    if W2 < 2: W2 = 2
    if H2 < 2: H2 = 2
    return W2, H2, s

def try_open_writer(target_path: str, fps: float, size: tuple[int,int], codec_pref: str = "auto"):
    """Try a list of codecs and container fallbacks until one opens."""
    W, H = size
    trials = []
    if codec_pref != "auto":
        trials = [(codec_pref, Path(target_path).suffix.lstrip("."))]
    else:
        # prefer H.264 in mp4, then mp4v, then AVI fallbacks
        trials = [("avc1","mp4"), ("H264","mp4"), ("mp4v","mp4"), ("XVID","avi"), ("MJPG","avi")]

    for fourcc_str, ext in trials:
        out_path = str(Path(target_path).with_suffix(f".{ext}"))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if writer.isOpened():
            print(f"[writer] using {fourcc_str} → {out_path} @ {W}x{H} {fps:.2f}fps")
            return writer, out_path
        else:
            writer.release()
    raise RuntimeError("Failed to open a VideoWriter with any supported codec/container.")

def ensure_even_size_frame(comp: np.ndarray) -> np.ndarray:
    """Crop by 1 px if needed to get even dims (some encoders require even)."""
    H, W = comp.shape[:2]
    H2, W2 = make_even(H), make_even(W)
    if (H2, W2) == (H, W):
        return comp
    return comp[:H2, :W2]


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Image path, directory of images, or video file")
    ap.add_argument("--target", required=True, help="Output image (.png) for single image, or video (.mp4) for multi-frame")
    ap.add_argument("--weights", default=None, help="(Optional) RF-DETR .pth local weights")
    ap.add_argument("--thr", type=float, default=0.5, help="Detection score threshold")
    ap.add_argument("--cols", type=int, default=None, help="Tiles per row (default: single row)")
    ap.add_argument("--k", type=int, default=6,
                    help="K-Means segments for unified PCA fusion (0 = auto from feature color separation)")

    ap.add_argument("--segment", action="store_true", help="Add unified segmentation tile")
    ap.add_argument("--classify", action="store_true", help="Add classes-color tile (segments -> COCO class_id mapping)")
    ap.add_argument("--class_method", choices=["agglom","kmeans"], default="agglom", help="Grouping method (if classify)")
    ap.add_argument("--class_k", type=int, default=None, help="Num classes if --class_method kmeans")
    ap.add_argument("--class_thresh", type=float, default=0.15, help="Agglomerative cosine distance threshold")
    ap.add_argument("--min_seg_px", type=int, default=64, help="Ignore tiny segments for grouping")
    ap.add_argument("--save_frames", action="store_true", help="Also save per-frame composite PNGs next to --target")
    ap.add_argument("--codec", default="auto",
                choices=["auto","avc1","H264","mp4v","XVID","MJPG"],
                help="Preferred codec; 'auto' tries H.264 then fallbacks.")
    ap.add_argument("--max_w", type=int, default=3840, help="Max composite width")
    ap.add_argument("--max_h", type=int, default=2160, help="Max composite height")
    ap.add_argument("--thickness", type=int, default=2, help="Box line thickness")

    args = ap.parse_args()

    # Model + hooks (do NOT optimize; we need hooks)
    wrapper = RFDETRMedium()
    if args.weights:
        raw = torch.load(args.weights, map_location="cpu")
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        wrapper.model.load_state_dict(state, strict=False)
    backbone = wrapper.model.model

    tapper = FeatureTapper(backbone); tapper.attach()
    box_annot = sv.BoxAnnotator(thickness=args.thickness); label_annot = sv.LabelAnnotator()
    pca_map = {}      # level -> PCAMapper
    final_keys = None

    # Input iterator (image, dir, or video)
    frames_iter = []
    single_image = False
    if os.path.isdir(args.source):
        files = list_images_in_dir(args.source)
        assert files, f"No images found in {args.source}"
        frames_iter = files
    elif is_video(args.source):
        frames_iter = [args.source]  # special-cased below
    elif is_image(args.source):
        frames_iter = [args.source]; single_image = True
    else:
        raise ValueError("source must be image, directory of images, or a video file")

    writer = None
    fps = 30.0
    comp_size = None
    frame_count = 0

    def process_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
        nonlocal final_keys, pca_map
        H, W = frame_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tapper.clear()
        dets = wrapper.predict(pil, threshold=args.thr)

        # Left tile: original + detections
        labels = [f"{COCO_CLASSES[c]} {p:.2f}" for c, p in zip(dets.class_id, dets.confidence)]
        left_bgr = ensure_bgr8(frame_bgr.copy())
        left_bgr = box_annotate(left_bgr, dets, box_annot, label_annot, labels)

        # Discover finals + fit PCA on first processed frame
        if final_keys is None:
            final_keys = pick_final_keys(tapper.captured)
            for k in final_keys:
                mapper = PCAMapper(); mapper.fit(tapper.captured[k]); pca_map[k] = mapper

        tiles = [left_bgr]

        # Build per-level PCA tiles with overlayed detections
        pca_rgbs = {}
        for k in final_keys:
            feat = tapper.captured.get(k, None)
            if feat is None: continue
            rgb01 = pca_map[k].transform(feat)      # Hf x Wf x 3 (0..1)
            pca_rgbs[k] = rgb01

            # 1) Make the feature tile the SAME SIZE as the original tile
            fmap_bgr = to_bgr_uint8(rgb01)
            fmap_bgr = cv2.resize(fmap_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
            fmap_bgr = ensure_bgr8(fmap_bgr)

            # 2) Draw boxes AFTER resize, using original dets (already in W,H coords)
            fmap_bgr = box_annotate(fmap_bgr, dets, box_annot, None, None)  # boxes only

            tiles.append(fmap_bgr)


        # Optional segmentation & classes-color panes
        if args.segment or args.classify:
            # decide k
            if args.k == 0:
                k_use, dbg = estimate_k_from_color_separation(
                    pca_rgbs, kmin=3, kmax=15, sample_px=30000, iters=20, seed=0, mode="max"
                )
                # (optional) print(f"[auto-k] sep={dbg['sep']:.3f} band={dbg['band']} chosen k={k_use}")
            else:
                k_use = max(2, int(args.k))

            # segment on COMMON FEATURE GRID with this k
            labels_feat, (Wc, Hc) = unified_segmentation_on_feature_grid(
                pca_rgbs=pca_rgbs, k=k_use, seed=0, mode="max"
            )
            # upsample IDs for display / mapping
            labels_img = np.array(
                Image.fromarray(labels_feat.astype(np.int32), mode="I")
                    .resize((W, H), resample=Image.NEAREST)
            )

            # show the segmentation (stable palette)
            seg_rgb = colorize_ids(labels_img)
            tiles.append(cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))

            if args.classify:
                rid2cls = map_regions_to_detection_classes(labels_img, dets, (W, H))
                class_rgb = color_map_from_class_ids(labels_img, rid2cls)
                tiles.append(cv2.cvtColor(class_rgb, cv2.COLOR_RGB2BGR))

        comp = tile_grid(tiles, tiles_per_row=args.cols)
        return comp

    # helpers used above
    def box_annotate(img_bgr, dets, box_annot, label_annot, labels):
        img_bgr = box_annot.annotate(img_bgr, dets)
        if label_annot is not None and labels is not None:
            img_bgr = label_annot.annotate(img_bgr, dets, labels)
        return img_bgr

    def colorize_segments_by_mean_image(labels: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
        H, W = labels.shape
        out = np.zeros((H, W, 3), np.uint8)
        for rid in np.unique(labels):
            m = (labels == rid)
            if m.any():
                out[m] = img_rgb[m].mean(axis=0).astype(np.uint8)
        return out

    # --------- Single image ---------
    if single_image:
        img = cv2.imread(frames_iter[0], cv2.IMREAD_COLOR)
        comp = process_frame_bgr(img)
        cv2.imwrite(args.target, comp)
        tapper.detach()
        print(f"[done] wrote {args.target}")
        return

    # --------- Directory of images ---------
    if os.path.isdir(args.source):
        files = frames_iter
        # Init writer after first frame (know composite size)
        first = cv2.imread(files[0], cv2.IMREAD_COLOR)
        comp = process_frame_bgr(first)
        # downscale + make even
        Hc, Wc = comp.shape[:2]
        Ww, Hw, _ = fit_size(Wc, Hc, args.max_w, args.max_h)
        if (Ww, Hw) != (Wc, Hc):
            comp = cv2.resize(comp, (Ww, Hw), interpolation=cv2.INTER_AREA)
        comp = ensure_even_size_frame(comp)

        # open writer with fallbacks
        fps = 30.0
        writer, out_path = try_open_writer(args.target, fps, (comp.shape[1], comp.shape[0]), codec_pref=args.codec)
        writer.write(comp)
        if args.save_frames:
            Path(args.target).with_suffix("")
        if args.save_frames:
            out_frames_dir = Path(args.target).with_suffix("").as_posix() + "_frames"
            Path(out_frames_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(out_frames_dir, f"frame_{0:06d}.png"), comp)
        
        target_size = (comp.shape[1], comp.shape[0])  # after first write
        for i, f in enumerate(files[1:], start=1):
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            comp = process_frame_bgr(img)
            if (comp.shape[1], comp.shape[0]) != target_size:
                comp = cv2.resize(comp, target_size, interpolation=cv2.INTER_AREA)
            comp = ensure_even_size_frame(comp)
            writer.write(comp)
            if args.save_frames:
                cv2.imwrite(os.path.join(out_frames_dir, f"frame_{i:06d}.png"), comp)
            if (i+1) % 50 == 0:
                print(f"[{i+1}/{len(files)}]")
        writer.release()
        tapper.detach()
        print(f"[done] wrote {args.target}")
        return

    # --------- Video file ---------
    if is_video(args.source):
        cap = cv2.VideoCapture(args.source)
        assert cap.isOpened(), f"Cannot open {args.source}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first_ok, first_frame = cap.read()
        assert first_ok, "Could not read first frame"
        comp = process_frame_bgr(first_frame)
        # downscale + make even
        Hc, Wc = comp.shape[:2]
        Ww, Hw, _ = fit_size(Wc, Hc, args.max_w, args.max_h)
        if (Ww, Hw) != (Wc, Hc):
            comp = cv2.resize(comp, (Ww, Hw), interpolation=cv2.INTER_AREA)
        comp = ensure_even_size_frame(comp)

        # open writer robustly
        writer, out_path = try_open_writer(args.target, fps, (comp.shape[1], comp.shape[0]), codec_pref=args.codec)
        writer.write(comp)
        target_size = (comp.shape[1], comp.shape[0])
        if args.save_frames:
            out_frames_dir = Path(args.target).with_suffix("").as_posix() + "_frames"
            Path(out_frames_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(out_frames_dir, f"frame_{0:06d}.png"), comp)

        idx = 1
        while True:
            ok, frame = cap.read()
            if not ok: break
            comp = process_frame_bgr(frame)
            if (comp.shape[1], comp.shape[0]) != target_size:
                comp = cv2.resize(comp, target_size, interpolation=cv2.INTER_AREA)
            comp = ensure_even_size_frame(comp)
            writer.write(comp)
            if args.save_frames:
                cv2.imwrite(os.path.join(out_frames_dir, f"frame_{idx:06d}.png"), comp)
            if idx % 50 == 0:
                print(f"[{idx}/{total if total else '?'}]")
            idx += 1
        cap.release(); writer.release(); tapper.detach()
        print(f"[done] wrote {args.target}")
        return


if __name__ == "__main__":
    main()
