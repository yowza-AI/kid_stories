@echo off
cd /d "%~dp0"
echo Installing PyTorch (CUDA 12.1 build) ...
py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo Installing qwen-tts + soundfile ...
py -m pip install -U qwen-tts soundfile
echo.
echo Done. The first story generated after this downloads the ~4.5GB
echo Qwen3-TTS 1.7B model (one time). It loads on GPU 1 (FP16) automatically.
