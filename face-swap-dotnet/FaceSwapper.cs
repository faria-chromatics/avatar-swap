using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceSwap;

/// <summary>
/// Face swap using InSwapper (inswapper_128.onnx) + ArcFace embedding (w600k_r50.onnx).
///
/// Pipeline:
///   1. Align source face → 112x112 → ArcFace → 512-dim embedding
///   2. Transform embedding via emap matrix
///   3. Align target face → 128x128
///   4. InSwapper(target_aligned, source_embedding) → swapped face 128x128
///   5. Paste back onto original image
/// </summary>
public class FaceSwapper : IDisposable
{
    private readonly InferenceSession _swapSession;
    private readonly InferenceSession _arcfaceSession;
    private readonly float[,] _emap; // 512x512 embedding transform matrix

    // ArcFace alignment template for 112x112
    private static readonly float[,] ArcFaceTemplate = {
        { 38.2946f, 51.6963f },
        { 73.5318f, 51.5014f },
        { 56.0252f, 71.7366f },
        { 41.5493f, 92.3655f },
        { 70.7299f, 92.2041f },
    };

    public FaceSwapper(string swapModelPath, string arcfaceModelPath, string emapPath)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        _swapSession = new InferenceSession(swapModelPath, opts);
        _arcfaceSession = new InferenceSession(arcfaceModelPath, opts);
        _emap = LoadEmap(emapPath);
    }

    /// <summary>Compute 512-dim ArcFace embedding for a face.</summary>
    public float[] GetEmbedding(Mat image, FaceInfo face)
    {
        // Align face to 112x112 using ArcFace template
        var aligned = ImageUtils.AlignFace(image, face.Landmarks, 112, ArcFaceTemplate);

        // Preprocess: BGR→RGB, (pixel - 127.5) / 127.5, HWC→CHW
        var blob = new float[3 * 112 * 112];
        var data = new byte[112 * 112 * 3];
        System.Runtime.InteropServices.Marshal.Copy(aligned.Data, data, 0, data.Length);

        for (int y = 0; y < 112; y++)
            for (int x = 0; x < 112; x++)
            {
                int src = (y * 112 + x) * 3;
                int pix = y * 112 + x;
                blob[0 * 112 * 112 + pix] = (data[src + 2] - 127.5f) / 127.5f;
                blob[1 * 112 * 112 + pix] = (data[src + 1] - 127.5f) / 127.5f;
                blob[2 * 112 * 112 + pix] = (data[src + 0] - 127.5f) / 127.5f;
            }

        aligned.Dispose();

        var tensor = new DenseTensor<float>(blob, new[] { 1, 3, 112, 112 });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input.1", tensor) };
        using var results = _arcfaceSession.Run(inputs);
        var embedding = results.First().AsEnumerable<float>().ToArray();

        // L2 normalize
        float norm = MathF.Sqrt(embedding.Sum(v => v * v));
        for (int i = 0; i < embedding.Length; i++) embedding[i] /= norm;

        return embedding;
    }

    /// <summary>Swap face: paste source identity onto target image.</summary>
    public Mat SwapFace(Mat image, FaceInfo targetFace, float[] sourceEmbedding)
    {
        // Scale template to 128x128
        float ratio = 128f / 112f;
        var template128 = new float[5, 2];
        for (int i = 0; i < 5; i++)
        {
            template128[i, 0] = ArcFaceTemplate[i, 0] * ratio;
            template128[i, 1] = ArcFaceTemplate[i, 1] * ratio;
        }

        // Align target face to 128x128
        var (aligned, M) = ImageUtils.AlignFaceWithMatrix(image, targetFace.Landmarks, 128, template128);

        // Preprocess aligned target: (pixel) / 255.0, BGR→RGB, HWC→CHW
        var targetBlob = new float[3 * 128 * 128];
        var data = new byte[128 * 128 * 3];
        System.Runtime.InteropServices.Marshal.Copy(aligned.Data, data, 0, data.Length);

        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 128; x++)
            {
                int src = (y * 128 + x) * 3;
                int pix = y * 128 + x;
                targetBlob[0 * 128 * 128 + pix] = data[src + 2] / 255f;
                targetBlob[1 * 128 * 128 + pix] = data[src + 1] / 255f;
                targetBlob[2 * 128 * 128 + pix] = data[src + 0] / 255f;
            }

        // Transform source embedding via emap
        var latent = new float[512];
        for (int i = 0; i < 512; i++)
        {
            float sum = 0;
            for (int j = 0; j < 512; j++)
                sum += sourceEmbedding[j] * _emap[j, i];
            latent[i] = sum;
        }
        // L2 normalize
        float norm = MathF.Sqrt(latent.Sum(v => v * v));
        for (int i = 0; i < 512; i++) latent[i] /= norm;

        // Run InSwapper
        var targetTensor = new DenseTensor<float>(targetBlob, new[] { 1, 3, 128, 128 });
        var sourceTensor = new DenseTensor<float>(latent, new[] { 1, 512 });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("target", targetTensor),
            NamedOnnxValue.CreateFromTensor("source", sourceTensor),
        };
        using var results = _swapSession.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        // Post-process: CHW→HWC, RGB→BGR, * 255, clip
        var swappedFace = new Mat(128, 128, MatType.CV_8UC3);
        var swappedData = new byte[128 * 128 * 3];
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 128; x++)
            {
                int pix = y * 128 + x;
                int dst = (y * 128 + x) * 3;
                swappedData[dst + 2] = (byte)Math.Clamp(output[0 * 128 * 128 + pix] * 255f, 0, 255); // R→B[2] (BGR)
                swappedData[dst + 1] = (byte)Math.Clamp(output[1 * 128 * 128 + pix] * 255f, 0, 255);
                swappedData[dst + 0] = (byte)Math.Clamp(output[2 * 128 * 128 + pix] * 255f, 0, 255);
            }
        System.Runtime.InteropServices.Marshal.Copy(swappedData, 0, swappedFace.Data, swappedData.Length);

        // Paste back onto original image with smooth elliptical blending
        var result = image.Clone();
        ImageUtils.PasteBackWithBlending(result, swappedFace, M, 128);

        aligned.Dispose();
        swappedFace.Dispose();
        return result;
    }

    private static float[,] LoadEmap(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);

        var emap = new float[512, 512];
        for (int i = 0; i < 512; i++)
            for (int j = 0; j < 512; j++)
                emap[i, j] = floats[i * 512 + j];
        return emap;
    }

    public void Dispose()
    {
        _swapSession.Dispose();
        _arcfaceSession.Dispose();
    }
}
