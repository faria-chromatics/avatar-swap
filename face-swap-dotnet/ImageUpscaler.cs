using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceSwap;

/// <summary>
/// Image upscaler using Real-ESRGAN x2 ONNX model.
/// Processes image in 64x64 tiles (model's fixed input size).
///
/// Model I/O:
///   Input:  "input" [1, 3, 64, 64]  — RGB, [0, 1]
///   Output: "output" [1, 3, 128, 128] — RGB, [0, 1]
/// </summary>
public class ImageUpscaler : IDisposable
{
    private readonly InferenceSession _session;
    private const int TileSize = 64;
    private const int Scale = 2;

    public ImageUpscaler(string modelPath)
    {
        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        _session = new InferenceSession(modelPath, opts);
    }

    /// <summary>Upscale an image by 2x using tile-based processing.</summary>
    public Mat Upscale(Mat image)
    {
        int h = image.Rows, w = image.Cols;

        // Pad to be divisible by TileSize
        int padH = (TileSize - h % TileSize) % TileSize;
        int padW = (TileSize - w % TileSize) % TileSize;

        var padded = new Mat();
        Cv2.CopyMakeBorder(image, padded, 0, padH, 0, padW, BorderTypes.Reflect);
        int pH = padded.Rows, pW = padded.Cols;

        int outH = pH * Scale, outW = pW * Scale;
        var output = new Mat(outH, outW, MatType.CV_8UC3);

        int totalTiles = (pH / TileSize) * (pW / TileSize);
        int tileNum = 0;

        // Read padded image data once
        var paddedData = new byte[pH * pW * 3];
        System.Runtime.InteropServices.Marshal.Copy(padded.Data, paddedData, 0, paddedData.Length);

        var outputData = new byte[outH * outW * 3];

        for (int y = 0; y < pH; y += TileSize)
        {
            for (int x = 0; x < pW; x += TileSize)
            {
                // Extract tile and preprocess: BGR→RGB, [0, 1], HWC→CHW
                var tileBlob = new float[3 * TileSize * TileSize];
                for (int ty = 0; ty < TileSize; ty++)
                    for (int tx = 0; tx < TileSize; tx++)
                    {
                        int srcIdx = ((y + ty) * pW + (x + tx)) * 3;
                        int pix = ty * TileSize + tx;
                        tileBlob[0 * TileSize * TileSize + pix] = paddedData[srcIdx + 2] / 255f; // R
                        tileBlob[1 * TileSize * TileSize + pix] = paddedData[srcIdx + 1] / 255f; // G
                        tileBlob[2 * TileSize * TileSize + pix] = paddedData[srcIdx + 0] / 255f; // B
                    }

                // Run model
                var tensor = new DenseTensor<float>(tileBlob, new[] { 1, 3, TileSize, TileSize });
                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", tensor) };
                using var results = _session.Run(inputs);
                var tileOut = results.First().AsEnumerable<float>().ToArray();

                int outTileH = TileSize * Scale, outTileW = TileSize * Scale;

                // Write tile output: CHW→HWC, RGB→BGR, [0,1]→[0,255]
                int oy = y * Scale, ox = x * Scale;
                for (int ty = 0; ty < outTileH; ty++)
                    for (int tx = 0; tx < outTileW; tx++)
                    {
                        int pix = ty * outTileW + tx;
                        int dstIdx = ((oy + ty) * outW + (ox + tx)) * 3;
                        outputData[dstIdx + 2] = (byte)Math.Clamp(tileOut[0 * outTileH * outTileW + pix] * 255f, 0, 255);
                        outputData[dstIdx + 1] = (byte)Math.Clamp(tileOut[1 * outTileH * outTileW + pix] * 255f, 0, 255);
                        outputData[dstIdx + 0] = (byte)Math.Clamp(tileOut[2 * outTileH * outTileW + pix] * 255f, 0, 255);
                    }

                tileNum++;
                if (tileNum % 50 == 0 || tileNum == totalTiles)
                    Console.WriteLine($"    Tile {tileNum}/{totalTiles}");
            }
        }

        System.Runtime.InteropServices.Marshal.Copy(outputData, 0, output.Data, outputData.Length);
        padded.Dispose();

        // Crop to remove padding
        var cropped = output[new Rect(0, 0, w * Scale, h * Scale)].Clone();
        output.Dispose();
        return cropped;
    }

    public void Dispose() => _session.Dispose();
}
