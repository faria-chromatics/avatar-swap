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

// ── Step 5: Detect face on swapped image ─────────────────────────────────────
Console.WriteLine("Detecting face on swapped result...");
var swappedFaces = detector.Detect(swapped);
if (swappedFaces.Count == 0)
{
    Console.WriteLine("  WARNING: No face detected after swap, saving raw result");
    Cv2.ImWrite($"{OutputDir}/result.jpg", swapped, new[] { (int)ImwriteFlags.JpegQuality, 95 });
    Console.WriteLine($"\nDone! Result saved to: {OutputDir}/result.jpg");
    sourceImg.Dispose(); targetImg.Dispose(); swapped.Dispose();
    return;
}

var swFace = swappedFaces.OrderByDescending(f => f.Area).First();

// ── Crop-based upscale — ESRGAN only the face region ─────────────────────────
Console.WriteLine("Cropping face region and upscaling with Real-ESRGAN...");
sw.Restart();

int x1 = (int)swFace.BBox[0], y1 = (int)swFace.BBox[1];
int x2 = (int)swFace.BBox[2], y2 = (int)swFace.BBox[3];
int faceW = x2 - x1, faceH = y2 - y1;
int pad = (int)(Math.Max(faceW, faceH) * 1.0);

int imgH = swapped.Rows, imgW = swapped.Cols;
int cropX1 = Math.Max(0, x1 - pad);
int cropY1 = Math.Max(0, y1 - pad);
int cropX2 = Math.Min(imgW, x2 + pad);
int cropY2 = Math.Min(imgH, y2 + pad);

var faceCrop = swapped[new Rect(cropX1, cropY1, cropX2 - cropX1, cropY2 - cropY1)].Clone();
var upscaledCrop = upscaler.Upscale(faceCrop);

sw.Stop();
Console.WriteLine($"  Crop upscale done in {sw.Elapsed.TotalSeconds:F1}s | " +
    $"{faceCrop.Rows}x{faceCrop.Cols} → {upscaledCrop.Rows}x{upscaledCrop.Cols}");
faceCrop.Dispose();

// ── Step 6: Detect face on upscaled crop for CodeFormer ──────────────────────
Console.WriteLine("Enhancing face with CodeFormer...");
sw.Restart();

var upCropFaces = detector.Detect(upscaledCrop);
if (upCropFaces.Count == 0)
{
    Console.WriteLine("  WARNING: No face detected in upscaled crop, saving raw result");
    Cv2.ImWrite($"{OutputDir}/result.jpg", swapped, new[] { (int)ImwriteFlags.JpegQuality, 95 });
    Console.WriteLine($"\nDone! Result saved to: {OutputDir}/result.jpg");
    sourceImg.Dispose(); targetImg.Dispose(); swapped.Dispose(); upscaledCrop.Dispose();
    return;
}

var upFace = upCropFaces.OrderByDescending(f => f.Area).First();
var resultCrop = enhancer.EnhanceFace(upscaledCrop, upFace);

sw.Stop();
Console.WriteLine($"  Face enhancement done in {sw.Elapsed.TotalSeconds:F1}s");

// ── Step 8: Paste enhanced crop back onto original-resolution image ──────────
Console.WriteLine("Compositing final result...");
var result = swapped.Clone();

int origCropH = cropY2 - cropY1;
int origCropW = cropX2 - cropX1;
var resultCropResized = new Mat();
Cv2.Resize(resultCrop, resultCropResized, new Size(origCropW, origCropH),
    interpolation: InterpolationFlags.Lanczos4);

// Soft feather at crop edges to avoid hard seams
const int blendMargin = 8;
var cropBlend = new Mat(origCropH, origCropW, MatType.CV_32FC1, Scalar.All(1.0));
var blendData = new float[origCropH * origCropW];
System.Runtime.InteropServices.Marshal.Copy(cropBlend.Data, blendData, 0, blendData.Length);

for (int r = 0; r < blendMargin; r++)
{
    float rampV = (float)r / blendMargin;
    float rampVInv = 1f - rampV;
    for (int c = 0; c < origCropW; c++)
    {
        // Top edge
        blendData[r * origCropW + c] = Math.Min(blendData[r * origCropW + c], rampV);
        // Bottom edge
        int br = origCropH - 1 - r;
        blendData[br * origCropW + c] = Math.Min(blendData[br * origCropW + c], rampV);
    }
}
for (int c = 0; c < blendMargin; c++)
{
    float rampH = (float)c / blendMargin;
    for (int r = 0; r < origCropH; r++)
    {
        // Left edge
        blendData[r * origCropW + c] = Math.Min(blendData[r * origCropW + c], rampH);
        // Right edge
        int rc = origCropW - 1 - c;
        blendData[r * origCropW + rc] = Math.Min(blendData[r * origCropW + rc], rampH);
    }
}

// Apply feathered blend
var regionData = new byte[origCropH * origCropW * 3];
var resizedData = new byte[origCropH * origCropW * 3];
var region = result[new Rect(cropX1, cropY1, origCropW, origCropH)].Clone(); // Clone for contiguous memory
System.Runtime.InteropServices.Marshal.Copy(region.Data, regionData, 0, regionData.Length);
System.Runtime.InteropServices.Marshal.Copy(resultCropResized.Data, resizedData, 0, resizedData.Length);

var blendedData = new byte[origCropH * origCropW * 3];
for (int i = 0; i < origCropH * origCropW; i++)
{
    float alpha = blendData[i];
    int idx = i * 3;
    blendedData[idx + 0] = (byte)(resizedData[idx + 0] * alpha + regionData[idx + 0] * (1f - alpha));
    blendedData[idx + 1] = (byte)(resizedData[idx + 1] * alpha + regionData[idx + 1] * (1f - alpha));
    blendedData[idx + 2] = (byte)(resizedData[idx + 2] * alpha + regionData[idx + 2] * (1f - alpha));
}

var blendedMat = new Mat(origCropH, origCropW, MatType.CV_8UC3);
System.Runtime.InteropServices.Marshal.Copy(blendedData, 0, blendedMat.Data, blendedData.Length);
blendedMat.CopyTo(result[new Rect(cropX1, cropY1, origCropW, origCropH)]);

Cv2.ImWrite($"{OutputDir}/result.jpg", result, new[] { (int)ImwriteFlags.JpegQuality, 95 });
Console.WriteLine($"\nDone! Result saved to: {OutputDir}/result.jpg");

// Cleanup
sourceImg.Dispose();
targetImg.Dispose();
swapped.Dispose();
upscaledCrop.Dispose();
resultCrop.Dispose();
resultCropResized.Dispose();
cropBlend.Dispose();
blendedMat.Dispose();
region.Dispose();
result.Dispose();
