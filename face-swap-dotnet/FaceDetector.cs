using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceSwap;

/// <summary>
/// SCRFD face detector using det_10g.onnx from InsightFace buffalo_l.
/// Detects faces and returns bounding boxes + 5-point landmarks.
///
/// Model I/O (det_10g, input 640x640):
///   Input:  "input.1" [1, 3, H, W]  — (pixel - 127.5) / 128.0, BGR→RGB
///   Outputs (9 total, 3 per stride [8, 16, 32]):
///     scores: [12800,1] [3200,1] [800,1]
///     bboxes: [12800,4] [3200,4] [800,4]
///     kps:    [12800,10] [3200,10] [800,10]
/// </summary>
public class FaceDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private const int DetSize = 640;
    private static readonly int[] Strides = { 8, 16, 32 };
    private const int NumAnchors = 2;

    public FaceDetector(string modelPath)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        _session = new InferenceSession(modelPath, opts);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public List<FaceInfo> Detect(Mat image, float threshold = 0.5f, float nmsThreshold = 0.4f)
    {
        int origH = image.Rows, origW = image.Cols;

        // Resize keeping aspect ratio, pad to DetSize x DetSize
        float scale = Math.Min((float)DetSize / origH, (float)DetSize / origW);
        int newW = (int)(origW * scale), newH = (int)(origH * scale);

        var resized = new Mat();
        Cv2.Resize(image, resized, new Size(newW, newH));

        var padded = new Mat(DetSize, DetSize, MatType.CV_8UC3, Scalar.All(0));
        resized.CopyTo(padded[new Rect(0, 0, newW, newH)]);

        // Preprocess: BGR→RGB, (pixel - 127.5) / 128.0, HWC→CHW
        var blob = PreprocessImage(padded);

        // Run inference
        var inputTensor = new DenseTensor<float>(blob, new[] { 1, 3, DetSize, DetSize });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };
        var results = _session.Run(inputs).ToList();

        // Parse outputs: 9 total — scores[0..2], bboxes[3..5], kps[6..8] per stride
        var allFaces = new List<FaceInfo>();

        for (int idx = 0; idx < 3; idx++)
        {
            int stride = Strides[idx];
            int featH = DetSize / stride, featW = DetSize / stride;

            var scores = results[idx + 0].AsEnumerable<float>().ToArray();  // 0,1,2
            var bboxes = results[idx + 3].AsEnumerable<float>().ToArray();  // 3,4,5
            var kps = results[idx + 6].AsEnumerable<float>().ToArray();     // 6,7,8

            // Generate anchor centers
            var anchors = GenerateAnchors(featH, featW, stride);

            int numAnchorsTotal = featH * featW * NumAnchors;
            for (int i = 0; i < numAnchorsTotal; i++)
            {
                if (scores[i] < threshold) continue;

                float cx = anchors[i * 2], cy = anchors[i * 2 + 1];

                var face = new FaceInfo { Score = scores[i] };

                // Decode bbox: distance from anchor center
                face.BBox[0] = (cx - bboxes[i * 4 + 0] * stride) / scale;
                face.BBox[1] = (cy - bboxes[i * 4 + 1] * stride) / scale;
                face.BBox[2] = (cx + bboxes[i * 4 + 2] * stride) / scale;
                face.BBox[3] = (cy + bboxes[i * 4 + 3] * stride) / scale;

                // Decode landmarks
                for (int j = 0; j < 5; j++)
                {
                    face.Landmarks[j, 0] = (cx + kps[i * 10 + j * 2] * stride) / scale;
                    face.Landmarks[j, 1] = (cy + kps[i * 10 + j * 2 + 1] * stride) / scale;
                }

                allFaces.Add(face);
            }
        }

        resized.Dispose();
        padded.Dispose();
        foreach (var r in results) r.Dispose();

        return Nms(allFaces, nmsThreshold);
    }

    private static float[] GenerateAnchors(int featH, int featW, int stride)
    {
        var anchors = new float[featH * featW * NumAnchors * 2];
        int idx = 0;
        for (int y = 0; y < featH; y++)
        {
            for (int x = 0; x < featW; x++)
            {
                float cx = x * stride, cy = y * stride;
                for (int a = 0; a < NumAnchors; a++)
                {
                    anchors[idx++] = cx;
                    anchors[idx++] = cy;
                }
            }
        }
        return anchors;
    }

    private static float[] PreprocessImage(Mat image)
    {
        var blob = new float[3 * DetSize * DetSize];
        var data = new byte[DetSize * DetSize * 3];
        System.Runtime.InteropServices.Marshal.Copy(image.Data, data, 0, data.Length);

        for (int y = 0; y < DetSize; y++)
        {
            for (int x = 0; x < DetSize; x++)
            {
                int srcIdx = (y * DetSize + x) * 3;
                int pixIdx = y * DetSize + x;
                // BGR→RGB and normalize
                blob[0 * DetSize * DetSize + pixIdx] = (data[srcIdx + 2] - 127.5f) / 128.0f; // R
                blob[1 * DetSize * DetSize + pixIdx] = (data[srcIdx + 1] - 127.5f) / 128.0f; // G
                blob[2 * DetSize * DetSize + pixIdx] = (data[srcIdx + 0] - 127.5f) / 128.0f; // B
            }
        }
        return blob;
    }

    private static List<FaceInfo> Nms(List<FaceInfo> faces, float threshold)
    {
        var sorted = faces.OrderByDescending(f => f.Score).ToList();
        var keep = new List<FaceInfo>();
        var suppressed = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (suppressed[i]) continue;
            keep.Add(sorted[i]);

            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j]) continue;
                if (IoU(sorted[i].BBox, sorted[j].BBox) > threshold)
                    suppressed[j] = true;
            }
        }
        return keep;
    }

    private static float IoU(float[] a, float[] b)
    {
        float x1 = Math.Max(a[0], b[0]), y1 = Math.Max(a[1], b[1]);
        float x2 = Math.Min(a[2], b[2]), y2 = Math.Min(a[3], b[3]);
        float inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return inter / (areaA + areaB - inter + 1e-6f);
    }

    public void Dispose() => _session.Dispose();
}
