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
    /// Enhance the face in the upscaled crop: Umeyama align → adaptive fidelity → CodeFormer →
    /// inverse warp → landmark mask → Poisson blend. Returns the blended crop.
    /// </summary>
    public Mat EnhanceFace(Mat upscaledCrop, FaceInfo face)
    {
        // Umeyama alignment to 512x512
        var (aligned, M) = ImageUtils.AlignFaceWithMatrix(upscaledCrop, face.Landmarks, 512, Template512);

        // Adaptive fidelity based on sharpness
        double fidelity = ImageUtils.AssessFaceQuality(aligned);
        Console.WriteLine($"  Adaptive fidelity: {fidelity:F2}");

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

        // Warp enhanced face back onto upscaled crop
        var MInv = new Mat();
        Cv2.InvertAffineTransform(M, MInv);
        int ucH = upscaledCrop.Rows, ucW = upscaledCrop.Cols;
        var warpedFace = new Mat();
        Cv2.WarpAffine(enhanced, warpedFace, MInv, new Size(ucW, ucH), InterpolationFlags.Linear);

        // Landmark-based adaptive mask
        var mask = ImageUtils.BuildLandmarkMask(face.Landmarks, ucH, ucW, expand: 1.8f);

        // Poisson blending (seamlessClone) replaces naive LAB color transfer
        var poissonMask = new Mat();
        Cv2.Threshold(mask, poissonMask, 128, 255, ThresholdTypes.Binary);
        poissonMask.ConvertTo(poissonMask, MatType.CV_8UC1);

        var moments = Cv2.Moments(poissonMask);
        Mat resultCrop;

        if (moments.M00 > 0)
        {
            int cx = (int)(moments.M10 / moments.M00);
            int cy = (int)(moments.M01 / moments.M00);
            try
            {
                resultCrop = new Mat();
                Cv2.SeamlessClone(warpedFace, upscaledCrop, poissonMask, new Point(cx, cy),
                    resultCrop, SeamlessCloneMethods.MixedClone);
            }
            catch
            {
                Console.WriteLine("  Poisson blending failed, falling back to alpha blend");
                resultCrop = AlphaBlendWithMask(warpedFace, upscaledCrop, mask);
            }
        }
        else
        {
            resultCrop = AlphaBlendWithMask(warpedFace, upscaledCrop, mask);
        }

        // Cleanup
        aligned.Dispose();
        M.Dispose();
        MInv.Dispose();
        enhanced.Dispose();
        warpedFace.Dispose();
        mask.Dispose();
        poissonMask.Dispose();

        return resultCrop;
    }

    private static Mat AlphaBlendWithMask(Mat foreground, Mat background, Mat mask8u)
    {
        int h = background.Rows, w = background.Cols;
        var result = new Mat(h, w, MatType.CV_8UC3);

        var fgData = new byte[h * w * 3];
        var bgData = new byte[h * w * 3];
        var mData = new byte[h * w];
        var outData = new byte[h * w * 3];

        System.Runtime.InteropServices.Marshal.Copy(foreground.Data, fgData, 0, fgData.Length);
        System.Runtime.InteropServices.Marshal.Copy(background.Data, bgData, 0, bgData.Length);
        System.Runtime.InteropServices.Marshal.Copy(mask8u.Data, mData, 0, mData.Length);

        for (int i = 0; i < h * w; i++)
        {
            float alpha = mData[i] / 255f;
            int idx = i * 3;
            outData[idx + 0] = (byte)(fgData[idx + 0] * alpha + bgData[idx + 0] * (1f - alpha));
            outData[idx + 1] = (byte)(fgData[idx + 1] * alpha + bgData[idx + 1] * (1f - alpha));
            outData[idx + 2] = (byte)(fgData[idx + 2] * alpha + bgData[idx + 2] * (1f - alpha));
        }

        System.Runtime.InteropServices.Marshal.Copy(outData, 0, result.Data, outData.Length);
        return result;
    }

    public void Dispose() => _session.Dispose();
}
