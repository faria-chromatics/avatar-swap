namespace FaceSwap;

/// <summary>Detected face with bounding box, 5 landmarks, and embedding.</summary>
public class FaceInfo
{
    /// <summary>Bounding box [x1, y1, x2, y2]</summary>
    public float[] BBox { get; set; } = new float[4];

    /// <summary>Detection confidence score</summary>
    public float Score { get; set; }

    /// <summary>5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth. Shape [5, 2]</summary>
    public float[,] Landmarks { get; set; } = new float[5, 2];

    /// <summary>512-dim ArcFace embedding (L2 normalized)</summary>
    public float[]? Embedding { get; set; }

    /// <summary>Face area in pixels (width * height)</summary>
    public float Area => (BBox[2] - BBox[0]) * (BBox[3] - BBox[1]);
}
