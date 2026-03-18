using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceSwap;

/// <summary>
/// Face enhancement using CodeFormer ONNX model.
///
/// Model I/O:
///   Input:  "input" [1, 3, 512, 512]  — RGB, normalized to [-1, 1]
///           "weight" scalar double    — fidelity (0.0=max enhance, 1.0=max identity)
///   Output: "output" [1, 3, 512, 512] — RGB, range [-1, 1]
/// </summary>
public class FaceEnhancer : IDisposable
{
    private readonly InferenceSession _session;

    // 5-point template for 512x512 CodeFormer alignment
    private static readonly float[,] Template512 = {
        { 192.98138f, 239.94708f },
        { 318.90277f, 240.19366f },
        { 256.63416f, 314.01935f },
        { 201.26117f, 371.41043f },
        { 313.08905f, 371.15118f },
    };

    public FaceEnhancer(string modelPath)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        _session = new InferenceSession(modelPath, opts);
    }

    /// <summary>
    /// Enhance the face in the image: align → CodeFormer → color correct → blend back.
    /// </summary>
    public void EnhanceFace(Mat image, FaceInfo face, double fidelity = 0.7)
    {
        // Align face to 512x512
        var (aligned, M) = ImageUtils.AlignFaceWithMatrix(image, face.Landmarks, 512, Template512);

        // Preprocess: BGR→RGB, normalize to [-1, 1], HWC→CHW
        var blob = new float[3 * 512 * 512];
        var data = new byte[512 * 512 * 3];
        System.Runtime.InteropServices.Marshal.Copy(aligned.Data, data, 0, data.Length);

        for (int y = 0; y < 512; y++)
            for (int x = 0; x < 512; x++)
            {
                int src = (y * 512 + x) * 3;
                int pix = y * 512 + x;
                blob[0 * 512 * 512 + pix] = (data[src + 2] / 255f - 0.5f) / 0.5f; // R
                blob[1 * 512 * 512 + pix] = (data[src + 1] / 255f - 0.5f) / 0.5f; // G
                blob[2 * 512 * 512 + pix] = (data[src + 0] / 255f - 0.5f) / 0.5f; // B
            }

        // Run CodeFormer
        var inputTensor = new DenseTensor<float>(blob, new[] { 1, 3, 512, 512 });
        var weightTensor = new DenseTensor<double>(new[] { fidelity }, ReadOnlySpan<int>.Empty);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor),
            NamedOnnxValue.CreateFromTensor("weight", weightTensor),
        };
        using var results = _session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Post-process: [-1,1] → [0,255], CHW→HWC, RGB→BGR
        var enhanced = new Mat(512, 512, MatType.CV_8UC3);
        var enhData = new byte[512 * 512 * 3];
        for (int y = 0; y < 512; y++)
            for (int x = 0; x < 512; x++)
            {
                int pix = y * 512 + x;
                int dst = (y * 512 + x) * 3;
                enhData[dst + 2] = (byte)Math.Clamp((output[0 * 512 * 512 + pix] + 1f) * 0.5f * 255f, 0, 255);
                enhData[dst + 1] = (byte)Math.Clamp((output[1 * 512 * 512 + pix] + 1f) * 0.5f * 255f, 0, 255);
                enhData[dst + 0] = (byte)Math.Clamp((output[2 * 512 * 512 + pix] + 1f) * 0.5f * 255f, 0, 255);
            }
        System.Runtime.InteropServices.Marshal.Copy(enhData, 0, enhanced.Data, enhData.Length);

        // Color-correct enhanced face to match original aligned crop
        ImageUtils.ColorTransfer(enhanced, aligned);

        // Paste back with elliptical blending
        ImageUtils.PasteBackWithBlending(image, enhanced, M, 512);

        aligned.Dispose();
        enhanced.Dispose();
    }

    public void Dispose() => _session.Dispose();
}
