using OpenCvSharp;

namespace FaceSwap;

/// <summary>Shared utilities: face alignment, blending, transforms.</summary>
public static class ImageUtils
{
    /// <summary>Align a face to target size using 5-point landmarks and a template.</summary>
    public static Mat AlignFace(Mat image, float[,] landmarks, int size, float[,] template)
    {
        var (aligned, _) = AlignFaceWithMatrix(image, landmarks, size, template);
        return aligned;
    }

    /// <summary>Align a face and return both the aligned image and the affine matrix (Umeyama similarity transform).</summary>
    public static (Mat aligned, Mat M) AlignFaceWithMatrix(Mat image, float[,] landmarks, int size, float[,] template)
    {
        var M = EstimateSimilarityTransform(landmarks, template);

        var aligned = new Mat();
        Cv2.WarpAffine(image, aligned, M, new Size(size, size), InterpolationFlags.Linear);

        return (aligned, M);
    }

    /// <summary>
    /// Umeyama similarity transform — proper 5-point alignment with
    /// rotation, uniform scale, and translation (no shear).
    /// </summary>
    public static Mat EstimateSimilarityTransform(float[,] srcPts, float[,] dstPts)
    {
        int num = srcPts.GetLength(0);

        // Compute means
        double srcMeanX = 0, srcMeanY = 0, dstMeanX = 0, dstMeanY = 0;
        for (int i = 0; i < num; i++)
        {
            srcMeanX += srcPts[i, 0]; srcMeanY += srcPts[i, 1];
            dstMeanX += dstPts[i, 0]; dstMeanY += dstPts[i, 1];
        }
        srcMeanX /= num; srcMeanY /= num;
        dstMeanX /= num; dstMeanY /= num;

        // Demean
        var srcDX = new double[num]; var srcDY = new double[num];
        var dstDX = new double[num]; var dstDY = new double[num];
        for (int i = 0; i < num; i++)
        {
            srcDX[i] = srcPts[i, 0] - srcMeanX;
            srcDY[i] = srcPts[i, 1] - srcMeanY;
            dstDX[i] = dstPts[i, 0] - dstMeanX;
            dstDY[i] = dstPts[i, 1] - dstMeanY;
        }

        // A = dst_demean.T @ src_demean / num  (2x2)
        double a00 = 0, a01 = 0, a10 = 0, a11 = 0;
        for (int i = 0; i < num; i++)
        {
            a00 += dstDX[i] * srcDX[i]; a01 += dstDX[i] * srcDY[i];
            a10 += dstDY[i] * srcDX[i]; a11 += dstDY[i] * srcDY[i];
        }
        a00 /= num; a01 /= num; a10 /= num; a11 /= num;

        double det = a00 * a11 - a01 * a10;
        double d0 = 1.0, d1 = det < 0 ? -1.0 : 1.0;

        // SVD of 2x2 matrix A
        var aMat = new Mat(2, 2, MatType.CV_64FC1);
        aMat.Set(0, 0, a00); aMat.Set(0, 1, a01);
        aMat.Set(1, 0, a10); aMat.Set(1, 1, a11);

        var w = new Mat(); var u = new Mat(); var vt = new Mat();
        Cv2.SVDecomp(aMat, w, u, vt);

        double s0 = w.At<double>(0, 0), s1 = w.At<double>(1, 0);

        // R = U @ diag(d) @ Vt
        double u00 = u.At<double>(0, 0) * d0, u01 = u.At<double>(0, 1) * d1;
        double u10 = u.At<double>(1, 0) * d0, u11 = u.At<double>(1, 1) * d1;
        double vt00 = vt.At<double>(0, 0), vt01 = vt.At<double>(0, 1);
        double vt10 = vt.At<double>(1, 0), vt11 = vt.At<double>(1, 1);

        double r00 = u00 * vt00 + u01 * vt10;
        double r01 = u00 * vt01 + u01 * vt11;
        double r10 = u10 * vt00 + u11 * vt10;
        double r11 = u10 * vt01 + u11 * vt11;

        // scale = sum(S * d) / src_var
        double srcVar = 0;
        for (int i = 0; i < num; i++)
            srcVar += srcDX[i] * srcDX[i] + srcDY[i] * srcDY[i];
        srcVar /= num;
        double scale = (s0 * d0 + s1 * d1) / (srcVar + 1e-8);

        // t = dst_mean - scale * R @ src_mean
        double tx = dstMeanX - scale * (r00 * srcMeanX + r01 * srcMeanY);
        double ty = dstMeanY - scale * (r10 * srcMeanX + r11 * srcMeanY);

        var M = new Mat(2, 3, MatType.CV_64FC1);
        M.Set(0, 0, scale * r00); M.Set(0, 1, scale * r01); M.Set(0, 2, tx);
        M.Set(1, 0, scale * r10); M.Set(1, 1, scale * r11); M.Set(1, 2, ty);

        aMat.Dispose(); w.Dispose(); u.Dispose(); vt.Dispose();
        return M;
    }

    /// <summary>Estimate sharpness via Laplacian variance and map to CodeFormer fidelity.</summary>
    public static double AssessFaceQuality(Mat faceCrop)
    {
        var gray = new Mat();
        Cv2.CvtColor(faceCrop, gray, ColorConversionCodes.BGR2GRAY);
        var laplacian = new Mat();
        Cv2.Laplacian(gray, laplacian, MatType.CV_64FC1);

        Cv2.MeanStdDev(laplacian, out _, out var stdDev);
        double lapVar = stdDev[0] * stdDev[0];

        gray.Dispose();
        laplacian.Dispose();

        if (lapVar > 200) return 0.9;
        if (lapVar > 100) return 0.7;
        if (lapVar > 50) return 0.5;
        return 0.3;
    }

    /// <summary>Build adaptive face mask from 5-point landmarks via expanded convex hull.</summary>
    public static Mat BuildLandmarkMask(float[,] kps, int height, int width, float expand = 1.8f)
    {
        // Center of landmarks
        float cx = 0, cy = 0;
        for (int i = 0; i < 5; i++) { cx += kps[i, 0]; cy += kps[i, 1]; }
        cx /= 5; cy /= 5;

        // Max distance from center
        float maxDist = 0;
        for (int i = 0; i < 5; i++)
        {
            float dx = kps[i, 0] - cx, dy = kps[i, 1] - cy;
            float dist = MathF.Sqrt(dx * dx + dy * dy);
            maxDist = Math.Max(maxDist, dist);
        }
        float radius = maxDist * expand;

        // Synthetic boundary around landmarks to form a face-shaped hull
        var allPts = new List<Point>();
        for (int i = 0; i < 5; i++)
            allPts.Add(new Point((int)kps[i, 0], (int)kps[i, 1]));

        for (int i = 0; i < 32; i++)
        {
            double angle = 2 * Math.PI * i / 32;
            allPts.Add(new Point(
                (int)(cx + radius * Math.Cos(angle)),
                (int)(cy + radius * Math.Sin(angle))));
        }

        var mask = new Mat(height, width, MatType.CV_8UC1, Scalar.All(0));
        var hull = Cv2.ConvexHull(allPts.ToArray());
        Cv2.FillConvexPoly(mask, hull, new Scalar(255));

        // Feather edges proportionally to face size
        int ksize = Math.Max((int)(radius * 0.4f) | 1, 21);
        Cv2.GaussianBlur(mask, mask, new Size(ksize, ksize), ksize * 0.3);

        return mask;
    }

    /// <summary>Paste a face back onto the original image using inverse affine transform with elliptical blending.</summary>
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
    public static void BlendWithMask(Mat target, Mat source, Mat mask)
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
}
