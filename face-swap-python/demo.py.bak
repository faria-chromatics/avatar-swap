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
TARGET_IMAGE = os.path.join(INPUT_DIR, "target.jpg")

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


def color_transfer(source, target):
    """Transfer color distribution from target to source in LAB space."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        src_mean, src_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std() + 1e-6
        tgt_mean, tgt_std = tgt_lab[:, :, ch].mean(), tgt_lab[:, :, ch].std() + 1e-6
        src_lab[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (tgt_std / src_std) + tgt_mean
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


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

    # ── Step 5: Upscale FULL image with Real-ESRGAN (x2) ─────────────────────
    print("Upscaling full image with Real-ESRGAN (this may take a minute on CPU)...")
    t = time.time()

    swapped_rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
    upscaled_rgb = esrgan_upscale(swapped_rgb, esrgan)
    upscaled = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
    print(f"  Upscale done in {time.time() - t:.1f}s | {swapped.shape[:2]} → {upscaled.shape[:2]}")

    # ── Step 6: CodeFormer face enhancement on upscaled image ─────────────────
    print("Enhancing face with CodeFormer...")
    t = time.time()

    upscaled_faces = face_analyser.get(upscaled)
    if not upscaled_faces:
        print("  WARNING: No face detected after upscale, saving upscaled result only")
        cv2.imwrite(output_path, upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\nDone! Result saved to: {output_path}")
        return

    up_face = max(upscaled_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    TEMPLATE_5 = np.array([
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(up_face.kps.astype(np.float32), TEMPLATE_5)
    aligned = cv2.warpAffine(upscaled, M, (512, 512), flags=cv2.INTER_LINEAR)

    # Run CodeFormer
    face_input = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face_input = (face_input - 0.5) / 0.5
    face_input = face_input.transpose(2, 0, 1)[np.newaxis, ...]
    fidelity = np.array([0.7], dtype=np.float64)

    enhanced = codeformer.run(None, {"input": face_input, "weight": fidelity})[0][0]
    enhanced = (enhanced + 1) * 0.5
    enhanced = enhanced.transpose(1, 2, 0)
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # Color-correct enhanced face to match upscaled image
    enhanced = color_transfer(enhanced, aligned)

    # ── Step 7: Paste enhanced face back onto upscaled image ──────────────────
    print("Blending enhanced face back...")
    h, w = upscaled.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    warped_face = cv2.warpAffine(enhanced, M_inv, (w, h), flags=cv2.INTER_LINEAR)

    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(mask, (256, 256), (210, 260), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 30).astype(np.float32) / 255.0

    warped_mask = cv2.warpAffine(mask, M_inv, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = np.clip(warped_mask, 0, 1)[:, :, np.newaxis]

    result = (warped_face * warped_mask + upscaled.astype(np.float32) * (1 - warped_mask)).astype(np.uint8)
    print(f"  Face enhancement done in {time.time() - t:.1f}s")

    # ── Step 8: Downscale back to original resolution ─────────────────────────
    orig_h, orig_w = target_img.shape[:2]
    result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nDone! Result saved to: {output_path}")


if __name__ == "__main__":
    main()
