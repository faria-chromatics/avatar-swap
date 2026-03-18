using System.Diagnostics;
using OpenCvSharp;
using FaceSwap;

// ── Paths ────────────────────────────────────────────────────────────────────
const string InputDir = "./input";
const string OutputDir = "./output";
const string SourceFacePath = $"{InputDir}/source_face.jpg";
const string TargetImagePath = $"{InputDir}/target.jpg";

const string ModelsDir = "./models";
const string DetModelPath = $"{ModelsDir}/det_10g.onnx";
const string ArcFaceModelPath = $"{ModelsDir}/w600k_r50.onnx";
const string SwapModelPath = $"{ModelsDir}/inswapper_128.onnx";
const string EmapPath = $"{ModelsDir}/emap.bin";
const string CodeFormerPath = $"{ModelsDir}/codeformer.onnx";
const string EsrganPath = $"{ModelsDir}/real_esrgan_x2.onnx";

// ── Validate ─────────────────────────────────────────────────────────────────
foreach (var (path, name) in new[] {
    (SourceFacePath, "Source face"), (TargetImagePath, "Target image"),
    (DetModelPath, "det_10g.onnx"), (ArcFaceModelPath, "w600k_r50.onnx"),
    (SwapModelPath, "inswapper_128.onnx"), (EmapPath, "emap.bin"),
    (CodeFormerPath, "codeformer.onnx"), (EsrganPath, "real_esrgan_x2.onnx"),
})
{
    if (!File.Exists(path))
    {
        Console.WriteLine($"ERROR: {name} not found at {path}");
        return;
    }
}

Directory.CreateDirectory(OutputDir);
Console.WriteLine("=== Face Swap (.NET, Local CPU) ===\n");
var sw = new Stopwatch();

// ── Step 1: Load models ──────────────────────────────────────────────────────
Console.WriteLine("Loading face detector (det_10g)...");
using var detector = new FaceDetector(DetModelPath);

Console.WriteLine("Loading face swapper (InSwapper + ArcFace)...");
using var swapper = new FaceSwapper(SwapModelPath, ArcFaceModelPath, EmapPath);

Console.WriteLine("Loading face enhancer (CodeFormer)...");
using var enhancer = new FaceEnhancer(CodeFormerPath);

Console.WriteLine("Loading image upscaler (Real-ESRGAN x2)...");
using var upscaler = new ImageUpscaler(EsrganPath);

// ── Step 2: Read images ──────────────────────────────────────────────────────
var sourceImg = Cv2.ImRead(SourceFacePath);
var targetImg = Cv2.ImRead(TargetImagePath);

if (sourceImg.Empty()) { Console.WriteLine($"ERROR: Could not read {SourceFacePath}"); return; }
if (targetImg.Empty()) { Console.WriteLine($"ERROR: Could not read {TargetImagePath}"); return; }

// ── Step 3: Detect faces ─────────────────────────────────────────────────────
Console.WriteLine("Detecting faces...");
var sourceFaces = detector.Detect(sourceImg);
if (sourceFaces.Count == 0) { Console.WriteLine("ERROR: No face detected in source image"); return; }

var targetFaces = detector.Detect(targetImg);
if (targetFaces.Count == 0) { Console.WriteLine("ERROR: No face detected in target image"); return; }

var sourceFace = sourceFaces.OrderByDescending(f => f.Area).First();
var targetFace = targetFaces.OrderByDescending(f => f.Area).First();
Console.WriteLine($"  Found {sourceFaces.Count} face(s) in source, {targetFaces.Count} in target");

// Get source embedding
Console.WriteLine("Computing face embedding...");
sourceFace.Embedding = swapper.GetEmbedding(sourceImg, sourceFace);

// ── Step 4: Swap face ────────────────────────────────────────────────────────
Console.WriteLine("Swapping face...");
sw.Restart();
var swapped = swapper.SwapFace(targetImg, targetFace, sourceFace.Embedding);
sw.Stop();
Console.WriteLine($"  Swap done in {sw.Elapsed.TotalSeconds:F1}s");

Cv2.ImWrite($"{OutputDir}/result_raw.jpg", swapped);
Console.WriteLine($"  Raw swap saved to: {OutputDir}/result_raw.jpg");

// ── Step 5: Upscale full image ───────────────────────────────────────────────
Console.WriteLine("Upscaling full image with Real-ESRGAN (this may take a minute on CPU)...");
sw.Restart();
var upscaled = upscaler.Upscale(swapped);
sw.Stop();
Console.WriteLine($"  Upscale done in {sw.Elapsed.TotalSeconds:F1}s | {swapped.Rows}x{swapped.Cols} → {upscaled.Rows}x{upscaled.Cols}");

// ── Step 6: Enhance face with CodeFormer ─────────────────────────────────────
Console.WriteLine("Enhancing face with CodeFormer...");
sw.Restart();

var upscaledFaces = detector.Detect(upscaled);
if (upscaledFaces.Count == 0)
{
    Console.WriteLine("  WARNING: No face detected after upscale, saving upscaled result only");
    Cv2.ImWrite($"{OutputDir}/result.jpg", upscaled, new[] { (int)ImwriteFlags.JpegQuality, 95 });
}
else
{
    var upFace = upscaledFaces.OrderByDescending(f => f.Area).First();
    enhancer.EnhanceFace(upscaled, upFace, fidelity: 0.7);
    sw.Stop();
    Console.WriteLine($"  Face enhancement done in {sw.Elapsed.TotalSeconds:F1}s");

    // ── Step 7: Downscale back to original resolution ────────────────────────
    var result = new Mat();
    Cv2.Resize(upscaled, result, new Size(targetImg.Cols, targetImg.Rows), interpolation: InterpolationFlags.Lanczos4);
    Cv2.ImWrite($"{OutputDir}/result.jpg", result, new[] { (int)ImwriteFlags.JpegQuality, 95 });
    result.Dispose();
}

Console.WriteLine($"\nDone! Result saved to: {OutputDir}/result.jpg");

// Cleanup
sourceImg.Dispose();
targetImg.Dispose();
swapped.Dispose();
upscaled.Dispose();
