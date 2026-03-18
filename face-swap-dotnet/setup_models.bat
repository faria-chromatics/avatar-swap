@echo off
echo Copying models to face-swap-dotnet\models\ ...

:: From buffalo_l (face detection + recognition)
copy "%USERPROFILE%\.insightface\models\buffalo_l\det_10g.onnx" models\
copy "%USERPROFILE%\.insightface\models\buffalo_l\w600k_r50.onnx" models\

:: From face-swap-python (swap, enhance, upscale, emap)
copy "..\face-swap-python\models\inswapper_128.onnx" models\
copy "..\face-swap-python\models\codeformer.onnx" models\
copy "..\face-swap-python\models\real_esrgan_x2.onnx" models\
copy "..\face-swap-python\models\emap.bin" models\

:: Copy test images
copy "..\face-swap-python\input\*" input\

echo Done!
dir models\
