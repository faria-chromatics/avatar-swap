using OpenCvSharp;

namespace FaceSwap;

/// <summary>Shared utilities: face alignment, blending, color transfer.</summary>
public static class ImageUtils
{
    /// <summary>Align a face to target size using 5-point landmarks and a template.</summary>
    public static Mat AlignFace(Mat image, float[,] landmarks, int size, float[,] template)
    {
        var (aligned, _) = AlignFaceWithMatrix(image, landmarks, size, template);
        return aligned;
    }

    /// <summary>Align a face and return both the aligned image and the affine matrix.</summary>
    public static (Mat aligned, Mat M) AlignFaceWithMatrix(Mat image, float[,] landmarks, int size, float[,] template)
    {
        var srcPts = new Point2f[5];
        var dstPts = new Point2f[5];
        for (int i = 0; i < 5; i++)
        {
            srcPts[i] = new Point2f(landmarks[i, 0], landmarks[i, 1]);
            dstPts[i] = new Point2f(template[i, 0], template[i, 1]);
        }

        // Use estimateAffinePartial2D for similarity transform (rotation + scale + translation)
        var srcMat = InputArray.Create(srcPts);
        var dstMat = InputArray.Create(dstPts);
        var M = Cv2.EstimateAffinePartial2D(srcMat, dstMat);

        var aligned = new Mat();
        Cv2.WarpAffine(image, aligned, M, new Size(size, size), InterpolationFlags.Linear);

        return (aligned, M);
    }

    /// <summary>Paste a face back onto the original image using inverse affine transform (hard paste).</summary>
    public static void PasteBack(Mat original, Mat face, Mat M, int faceSize)
    {
        var MInv = new Mat();
        Cv2.InvertAffineTransform(M, MInv);

        int h = original.Rows, w = original.Cols;
        var warped = new Mat();
        Cv2.WarpAffine(face, warped, MInv, new Size(w, h), InterpolationFlags.Linear);

        // Create mask and warp it
        var mask = new Mat(faceSize, faceSize, MatType.CV_8UC1, new Scalar(255));
        var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(7, 7));
        Cv2.Erode(mask, mask, kernel, iterations: 3);
        Cv2.GaussianBlur(mask, mask, new Size(21, 21), 11);

        var warpedMask = new Mat();
        Cv2.WarpAffine(mask, warpedMask, MInv, new Size(w, h), InterpolationFlags.Linear);

        // Blend using mask
        BlendWithMask(original, warped, warpedMask);

        MInv.Dispose();
        warped.Dispose();
        mask.Dispose();
        kernel.Dispose();
        warpedMask.Dispose();
    }

    /// <summary>Paste enhanced face back with elliptical mask and smooth blending.</summary>
    public static void PasteBackWithBlending(Mat original, Mat face, Mat M, int faceSize)
    {
        var MInv = new Mat();
        Cv2.InvertAffineTransform(M, MInv);

        int h = original.Rows, w = original.Cols;
        var warped = new Mat();
        Cv2.WarpAffine(face, warped, MInv, new Size(w, h), InterpolationFlags.Linear);

        // Elliptical mask for natural face-shaped blending (scaled to faceSize)
        int axisX = (int)(faceSize * 0.41); // 210/512 ≈ 0.41
        int axisY = (int)(faceSize * 0.51); // 260/512 ≈ 0.51
        int blurK = Math.Max(3, faceSize / 10) | 1; // must be odd
        var mask = new Mat(faceSize, faceSize, MatType.CV_8UC1, new Scalar(0));
        Cv2.Ellipse(mask, new Point(faceSize / 2, faceSize / 2), new Size(axisX, axisY),
                    0, 0, 360, new Scalar(255), -1);
        Cv2.GaussianBlur(mask, mask, new Size(blurK, blurK), blurK / 3.0);

        var warpedMask = new Mat();
        Cv2.WarpAffine(mask, warpedMask, MInv, new Size(w, h), InterpolationFlags.Linear);

        BlendWithMask(original, warped, warpedMask);

        MInv.Dispose();
        warped.Dispose();
        mask.Dispose();
        warpedMask.Dispose();
    }

    /// <summary>Alpha blend source onto target using a grayscale mask (in-place on target).</summary>
    private static void BlendWithMask(Mat target, Mat source, Mat mask)
    {
        int h = target.Rows, w = target.Cols;
        var tData = new byte[h * w * 3];
        var sData = new byte[h * w * 3];
        var mData = new byte[h * w];

        System.Runtime.InteropServices.Marshal.Copy(target.Data, tData, 0, tData.Length);
        System.Runtime.InteropServices.Marshal.Copy(source.Data, sData, 0, sData.Length);
        System.Runtime.InteropServices.Marshal.Copy(mask.Data, mData, 0, mData.Length);

        for (int i = 0; i < h * w; i++)
        {
            float alpha = mData[i] / 255f;
            if (alpha < 0.001f) continue;

            int idx = i * 3;
            tData[idx + 0] = (byte)(sData[idx + 0] * alpha + tData[idx + 0] * (1f - alpha));
            tData[idx + 1] = (byte)(sData[idx + 1] * alpha + tData[idx + 1] * (1f - alpha));
            tData[idx + 2] = (byte)(sData[idx + 2] * alpha + tData[idx + 2] * (1f - alpha));
        }

        System.Runtime.InteropServices.Marshal.Copy(tData, 0, target.Data, tData.Length);
    }

    /// <summary>Transfer color distribution from target to source in LAB space (modifies source in-place).</summary>
    public static void ColorTransfer(Mat source, Mat target)
    {
        var srcLab = new Mat();
        var tgtLab = new Mat();
        Cv2.CvtColor(source, srcLab, ColorConversionCodes.BGR2Lab);
        Cv2.CvtColor(target, tgtLab, ColorConversionCodes.BGR2Lab);

        var srcF = new Mat();
        var tgtF = new Mat();
        srcLab.ConvertTo(srcF, MatType.CV_32FC3);
        tgtLab.ConvertTo(tgtF, MatType.CV_32FC3);

        for (int ch = 0; ch < 3; ch++)
        {
            var srcCh = new Mat();
            var tgtCh = new Mat();
            Cv2.ExtractChannel(srcF, srcCh, ch);
            Cv2.ExtractChannel(tgtF, tgtCh, ch);

            Cv2.MeanStdDev(srcCh, out var srcMean, out var srcStd);
            Cv2.MeanStdDev(tgtCh, out var tgtMean, out var tgtStd);

            double sm = srcMean[0], ss = srcStd[0] + 1e-6;
            double tm = tgtMean[0], ts = tgtStd[0] + 1e-6;

            // (pixel - src_mean) * (tgt_std / src_std) + tgt_mean
            var temp = new Mat();
            Cv2.Subtract(srcCh, new Scalar(sm), temp);
            Cv2.Multiply(temp, new Scalar(ts / ss), temp);
            Cv2.Add(temp, new Scalar(tm), temp);
            Cv2.InsertChannel(temp, srcF, ch);

            srcCh.Dispose();
            tgtCh.Dispose();
            temp.Dispose();
        }

        srcF.ConvertTo(srcLab, MatType.CV_8UC3);
        Cv2.CvtColor(srcLab, source, ColorConversionCodes.Lab2BGR);

        srcLab.Dispose();
        tgtLab.Dispose();
        srcF.Dispose();
        tgtF.Dispose();
    }
}
