"""
Face swap demo — local InsightFace + CodeFormer + Real-ESRGAN

Usage:
    python demo.py

Reads:
    input/source_face.jpg   — the face to swap in (avatar)
    input/target.jpg        — the image whose face gets replaced

Saves:
    output/result.jpg       — final enhanced result
    output/result_raw.jpg   — raw swap (before enhancement)
"""

import os
import sys
import time

import cv2
import numpy as np
import insightface
import onnxruntime

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

SOURCE_FACE = os.path.join(INPUT_DIR, "source_face.jpg")
TARGET_IMAGE = os.path.join(INPUT_DIR, "target.webp")

ONNX_MODEL = "./models/inswapper_128.onnx"
CODEFORMER_MODEL = "./models/codeformer.onnx"
ESRGAN_MODEL = "./models/real_esrgan_x2.onnx"


def esrgan_upscale(img, session, tile_size=64):
    """Tile-based Real-ESRGAN x2 upscale. Model expects 64x64 tiles."""
    h, w = img.shape[:2]
    scale = 2

    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]

    out_h, out_w = ph * scale, pw * scale
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)

    total_tiles = (ph // tile_size) * (pw // tile_size)
    tile_num = 0

    for y in range(0, ph, tile_size):
        for x in range(0, pw, tile_size):
            tile = padded[y:y + tile_size, x:x + tile_size]
            tile_input = tile.astype(np.float32) / 255.0
            tile_input = tile_input.transpose(2, 0, 1)[np.newaxis, ...]

            tile_output = session.run(None, {"input": tile_input})[0][0]
            tile_output = tile_output.transpose(1, 2, 0)
            tile_output = np.clip(tile_output * 255.0, 0, 255)

            oy, ox = y * scale, x * scale
            output[oy:oy + tile_output.shape[0], ox:ox + tile_output.shape[1]] = tile_output

            tile_num += 1
            if tile_num % 50 == 0 or tile_num == total_tiles:
                print(f"    Tile {tile_num}/{total_tiles}")

    output = output[:h * scale, :w * scale]
    return output.astype(np.uint8)


# ── Fix #5: Umeyama similarity transform replaces estimateAffinePartial2D ─────
def estimate_similarity_transform(src_pts, dst_pts):
    """
    Umeyama similarity transform — proper 5-point alignment with
    rotation, uniform scale, and translation (no shear).
    """
    num = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)

    src_demean = src_pts - src_mean
    dst_demean = dst_pts - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones(2)
    if np.linalg.det(A) < 0:
        d[1] = -1

    U, S, Vt = np.linalg.svd(A)
    R = U @ np.diag(d) @ Vt

    src_var = np.mean(np.sum(src_demean ** 2, axis=1))
    scale = np.sum(S * d) / (src_var + 1e-8)

    t = dst_mean - scale * R @ src_mean

    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M


# ── Fix #4: Adaptive fidelity based on sharpness assessment ──────────────────
def assess_face_quality(face_crop):
    """Estimate sharpness via Laplacian variance → map to CodeFormer fidelity."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Sharp → high fidelity (preserve), blurry → low fidelity (generate)
    if lap_var > 200:
        return 0.9
    elif lap_var > 100:
        return 0.7
    elif lap_var > 50:
        return 0.5
    else:
        return 0.3


# ── Fix #2: Landmark-based mask replaces hardcoded ellipse ────────────────────
def build_landmark_mask(kps, img_shape, expand=1.8):
    """Build adaptive face mask from 5-point landmarks via expanded convex hull."""
    h, w = img_shape[:2]

    center = kps.mean(axis=0)
    dists = np.linalg.norm(kps - center, axis=1)
    radius = dists.max() * expand

    # Synthetic boundary around landmarks to form a face-shaped hull
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    boundary = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    all_pts = np.vstack([kps, boundary]).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(all_pts)
    cv2.fillConvexPoly(mask, hull, 255)

    # Feather edges proportionally to face size
    ksize = max(int(radius * 0.4) | 1, 21)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize * 0.3)

    return mask.astype(np.float32) / 255.0


def main():
    # Validate inputs
    for path, label in [(SOURCE_FACE, "Source face"), (TARGET_IMAGE, "Target image")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            print(f"  Place your images in the '{INPUT_DIR}/' folder")
            sys.exit(1)

    for path, name in [
        (ONNX_MODEL, "inswapper_128.onnx"),
        (CODEFORMER_MODEL, "codeformer.onnx"),
        (ESRGAN_MODEL, "real_esrgan_x2.onnx"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "result.jpg")

    print("=== Face Swap (Local CPU) ===\n")

    # ── Step 1: Load models ───────────────────────────────────────────────────
    print("Loading face detection model (buffalo_l)...")
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading swap model (inswapper_128)...")
    swapper = insightface.model_zoo.get_model(
        ONNX_MODEL,
        providers=["CPUExecutionProvider"],
    )

    print("Loading face enhancement model (CodeFormer)...")
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    codeformer = onnxruntime.InferenceSession(
        CODEFORMER_MODEL, sess_options=opts, providers=["CPUExecutionProvider"],
    )

    print("Loading image upscaler (Real-ESRGAN x2)...")
    esrgan = onnxruntime.InferenceSession(
        ESRGAN_MODEL, sess_options=opts, providers=["CPUExecutionProvider"],
    )

    # ── Step 2: Read images ───────────────────────────────────────────────────
    source_img = cv2.imread(SOURCE_FACE)
    target_img = cv2.imread(TARGET_IMAGE)

    if source_img is None:
        print(f"ERROR: Could not read {SOURCE_FACE}")
        sys.exit(1)
    if target_img is None:
        print(f"ERROR: Could not read {TARGET_IMAGE}")
        sys.exit(1)

    # ── Step 3: Detect faces ──────────────────────────────────────────────────
    print("Detecting faces...")
    source_faces = face_analyser.get(source_img)
    if not source_faces:
        print("ERROR: No face detected in source image")
        sys.exit(1)

    target_faces = face_analyser.get(target_img)
    if not target_faces:
        print("ERROR: No face detected in target image")
        sys.exit(1)

    source_face = max(source_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    target_face = max(target_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"  Found {len(source_faces)} face(s) in source, {len(target_faces)} in target")

    # ── Step 4: Swap face (128x128 — raw) ─────────────────────────────────────
    print("Swapping face...")
    t = time.time()
    swapped = swapper.get(target_img, target_face, source_face, paste_back=True)
    print(f"  Swap done in {time.time() - t:.1f}s")

    raw_path = os.path.join(OUTPUT_DIR, "result_raw.jpg")
    cv2.imwrite(raw_path, swapped)
    print(f"  Raw swap saved to: {raw_path}")

    # ── Step 5: Detect face on swapped image ──────────────────────────────────
    print("Detecting face on swapped result...")
    swapped_faces = face_analyser.get(swapped)
    if not swapped_faces:
        print("  WARNING: No face detected after swap, saving raw result")
        cv2.imwrite(output_path, swapped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\nDone! Result saved to: {output_path}")
        return

    sw_face = max(swapped_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ── Fix #1: Crop-based upscale — ESRGAN only the face region ─────────────
    print("Cropping face region and upscaling with Real-ESRGAN...")
    t = time.time()

    x1, y1, x2, y2 = sw_face.bbox.astype(int)
    face_w, face_h = x2 - x1, y2 - y1
    pad = int(max(face_w, face_h) * 1.0)

    img_h, img_w = swapped.shape[:2]
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(img_w, x2 + pad)
    crop_y2 = min(img_h, y2 + pad)

    face_crop = swapped[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    upscaled_crop_rgb = esrgan_upscale(crop_rgb, esrgan)
    upscaled_crop = cv2.cvtColor(upscaled_crop_rgb, cv2.COLOR_RGB2BGR)

    print(f"  Crop upscale done in {time.time() - t:.1f}s | "
          f"{face_crop.shape[:2]} → {upscaled_crop.shape[:2]}")

    # ── Step 6: Detect face on upscaled crop for CodeFormer ───────────────────
    print("Enhancing face with CodeFormer...")
    t = time.time()

    up_crop_faces = face_analyser.get(upscaled_crop)
    if not up_crop_faces:
        print("  WARNING: No face detected in upscaled crop, saving raw result")
        cv2.imwrite(output_path, swapped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\nDone! Result saved to: {output_path}")
        return

    up_face = max(up_crop_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ── Fix #5: Umeyama 5-point similarity transform ─────────────────────────
    TEMPLATE_5 = np.array([
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ], dtype=np.float64)

    M = estimate_similarity_transform(up_face.kps.astype(np.float64), TEMPLATE_5)
    aligned = cv2.warpAffine(upscaled_crop, M, (512, 512), flags=cv2.INTER_LINEAR)

    # ── Fix #4: Adaptive fidelity ────────────────────────────────────────────
    fidelity_val = assess_face_quality(aligned)
    print(f"  Adaptive fidelity: {fidelity_val:.2f}")

    # Run CodeFormer
    face_input = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face_input = (face_input - 0.5) / 0.5
    face_input = face_input.transpose(2, 0, 1)[np.newaxis, ...]
    fidelity = np.array([fidelity_val], dtype=np.float64)

    enhanced = codeformer.run(None, {"input": face_input, "weight": fidelity})[0][0]
    enhanced = (enhanced + 1) * 0.5
    enhanced = enhanced.transpose(1, 2, 0)
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # ── Step 7: Warp enhanced face back onto upscaled crop ────────────────────
    M_inv = cv2.invertAffineTransform(M)
    uc_h, uc_w = upscaled_crop.shape[:2]
    warped_face = cv2.warpAffine(enhanced, M_inv, (uc_w, uc_h), flags=cv2.INTER_LINEAR)

    # ── Fix #2: Landmark-based adaptive mask ─────────────────────────────────
    mask = build_landmark_mask(up_face.kps, upscaled_crop.shape, expand=1.8)
    mask_3ch = mask[:, :, np.newaxis]

    # ── Fix #3: Poisson blending replaces naive LAB color transfer ────────────
    poisson_mask = (mask * 255).astype(np.uint8)
    poisson_mask = cv2.threshold(poisson_mask, 128, 255, cv2.THRESH_BINARY)[1]

    moments = cv2.moments(poisson_mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        try:
            result_crop = cv2.seamlessClone(
                warped_face, upscaled_crop, poisson_mask, (cx, cy), cv2.MIXED_CLONE,
            )
        except cv2.error:
            print("  Poisson blending failed, falling back to alpha blend")
            result_crop = (
                warped_face * mask_3ch
                + upscaled_crop.astype(np.float32) * (1 - mask_3ch)
            ).astype(np.uint8)
    else:
        result_crop = (
            warped_face * mask_3ch
            + upscaled_crop.astype(np.float32) * (1 - mask_3ch)
        ).astype(np.uint8)

    print(f"  Face enhancement done in {time.time() - t:.1f}s")

    # ── Step 8: Paste enhanced crop back onto original-resolution image ───────
    print("Compositing final result...")
    result = swapped.copy()

    orig_crop_h = crop_y2 - crop_y1
    orig_crop_w = crop_x2 - crop_x1
    result_crop_resized = cv2.resize(
        result_crop, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_LANCZOS4,
    )

    # Soft feather at crop edges to avoid hard seams
    blend_margin = 8
    crop_blend = np.ones((orig_crop_h, orig_crop_w), dtype=np.float32)
    ramp_v = np.linspace(0, 1, blend_margin)
    ramp_h = np.linspace(0, 1, blend_margin)

    crop_blend[:blend_margin, :] = np.minimum(
        crop_blend[:blend_margin, :], ramp_v[:, np.newaxis],
    )
    crop_blend[-blend_margin:, :] = np.minimum(
        crop_blend[-blend_margin:, :], ramp_v[::-1, np.newaxis],
    )
    crop_blend[:, :blend_margin] = np.minimum(
        crop_blend[:, :blend_margin], ramp_h[np.newaxis, :],
    )
    crop_blend[:, -blend_margin:] = np.minimum(
        crop_blend[:, -blend_margin:], ramp_h[np.newaxis, ::-1],
    )
    crop_blend = crop_blend[:, :, np.newaxis]

    region = result[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
    blended = (
        result_crop_resized.astype(np.float32) * crop_blend
        + region * (1 - crop_blend)
    ).astype(np.uint8)
    result[crop_y1:crop_y2, crop_x1:crop_x2] = blended

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nDone! Result saved to: {output_path}")


if __name__ == "__main__":
    main()