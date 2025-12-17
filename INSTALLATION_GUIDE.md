# Installation & Troubleshooting Guide

## Pre-Installation Checklist

- [ ] Python 3.11 or higher installed (`python --version`)
- [ ] Git installed for cloning repository
- [ ] 16GB+ RAM available
- [ ] 20GB+ free disk space
- [ ] Internet connection for downloading models
- [ ] Virtual environment tool (venv or conda)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/arturwyroslak/qwenedit.git
cd qwenedit
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python3.11 -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n qwenedit python=3.11
conda activate qwenedit
```

### 3. Install PyTorch

Choose the appropriate command based on your system:

**CPU-only (recommended for most users):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**NVIDIA GPU (CUDA 12.4):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Mac with Metal (M1/M2/M3):**
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python cli.py info
```

Expected output:
```
System Information:
  Device: cpu (or cuda/mps if available)
  Precision: torch.bfloat16
  CUDA Available: False (or True if GPU detected)
  PyTorch Version: 2.7.0
```

## Running the Application

### Option 1: Gradio Web Interface (Recommended)

```bash
python app.py
```

Open browser and navigate to: `http://localhost:7860`

### Option 2: Command Line

```bash
# Single transfer
python cli.py transfer source.jpg target.jpg -o output.jpg

# Batch generation
python cli.py batch source.jpg target.jpg -o outputs/ --count 3
```

## Troubleshooting

### Issue: ModuleNotFoundError - diffusers

**Problem:** `ModuleNotFoundError: No module named 'diffusers'`

**Solution:**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Issue: Out of Memory (OOM)

**Problem:** CUDA out of memory or insufficient RAM

**Solutions:**

1. Reduce image resolution:
```bash
python cli.py transfer source.jpg target.jpg --steps 20 -o output.jpg
```

2. Use CPU instead of GPU:
```bash
export DEVICE=cpu
python app.py
```

3. Close other applications to free up RAM

4. Reduce inference steps:
```bash
python cli.py transfer source.jpg target.jpg --steps 20 -o output.jpg
```

### Issue: Model Download Fails

**Problem:** HuggingFace model download times out or fails

**Solutions:**

1. Check internet connection
```bash
python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co')"
```

2. Set custom HuggingFace cache directory:
```bash
export HF_HOME=/path/to/custom/cache
export HF_TOKEN=your_token  # if needed
```

3. Clear cache and retry:
```bash
rm -rf ~/.cache/huggingface
python app.py  # Will re-download models
```

4. Manually download models:
```python
from diffusers import QwenImageEditPipeline
import torch

model = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)
model.save_pretrained("./qwen-model")
```

### Issue: Very Slow Performance

**Problem:** Generation takes 10+ minutes on CPU

**Expected:** 
- CPU: 2-3 minutes for 512x512 with 40 steps
- GPU: 15-20 seconds

**Solutions:**

1. Use GPU if available:
```bash
python cli.py info  # Check if CUDA detected
export DEVICE=cuda
python app.py
```

2. Reduce image resolution and steps:
```bash
python cli.py transfer source.jpg target.jpg --steps 20 -o output.jpg
```

3. Reduce guidance scale:
```bash
python cli.py transfer source.jpg target.jpg --cfg 2.0 -o output.jpg
```

### Issue: Poor Output Quality

**Problem:** Generated image looks blurry or distorted

**Solutions:**

1. Increase inference steps:
```bash
python cli.py transfer source.jpg target.jpg --steps 50 -o output.jpg
```

2. Increase guidance scale:
```bash
python cli.py transfer source.jpg target.jpg --cfg 5.0 -o output.jpg
```

3. Use higher quality input images

4. Improve prompt specificity

5. Try different seeds:
```bash
python cli.py transfer source.jpg target.jpg --seed 42 -o output.jpg
```

### Issue: Port 7860 Already in Use

**Problem:** `Address already in use` when starting Gradio

**Solutions:**

1. Use different port:
```bash
export GRADIO_SERVER_PORT=7861
python app.py
```

2. Kill process using the port:
```bash
# Linux/Mac
lsof -i :7860
kill -9 <PID>

# Windows
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

### Issue: Image Upload Fails

**Problem:** "File upload failed" in Gradio UI

**Solutions:**

1. Check file size (should be < 10MB)
```bash
ls -lh image.jpg
```

2. Check file format (use JPG, PNG, or BMP)
```bash
file image.jpg
```

3. Convert if needed:
```bash
ffmpeg -i image.jpg -q:v 2 image_converted.jpg
```

### Issue: "torch.cuda.OutOfMemoryError"

**Problem:** CUDA out of memory even with GPU

**Solutions:**

1. Reduce batch size (already sequential)

2. Use CPU for processing:
```bash
export DEVICE=cpu
python app.py
```

3. Clear CUDA cache between runs (already done)

4. Reduce image resolution:
```python
# In app, reduce max_size parameter
app.preprocess_image(image, max_size=512)  # or 256
```

### Issue: CUDA Version Mismatch

**Problem:** `CUDA driver version is insufficient`

**Solution:**

1. Check CUDA version:
```bash
nvidia-smi
```

2. Install compatible PyTorch:
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: macOS Metal Performance Issues

**Problem:** GPU acceleration not working on Mac

**Solutions:**

1. Verify MPS support:
```python
import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

2. Reinstall with conda:
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

3. Force CPU mode:
```bash
export DEVICE=cpu
python app.py
```

## Performance Optimization

### For CPU-only Systems

```bash
# Use smaller batch, fewer steps
python cli.py transfer source.jpg target.jpg --steps 20 -o output.jpg

# Set optimal environment
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4
python app.py
```

### For GPUs with Limited VRAM

```bash
# Reduce image size, use attention slicing
export CUDA_VISIBLE_DEVICES=0
python cli.py transfer source.jpg target.jpg --steps 20 -o output.jpg
```

### For Multi-GPU Systems

```bash
# Use first GPU
export CUDA_VISIBLE_DEVICES=0
python app.py

# Distribute workload
export CUDA_VISIBLE_DEVICES=0,1,2,3
python app.py
```

## Environment Variables Reference

```bash
# Device Selection
export DEVICE=cpu              # Force CPU
export DEVICE=cuda             # Force CUDA GPU
export DEVICE=mps              # Force Metal Performance Shaders (Mac)
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# Server Configuration
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860
export GRADIO_SHARE=false

# HuggingFace Configuration
export HF_HOME=/path/to/cache
export HF_TOKEN=your_token

# PyTorch Configuration
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export TORCH_HOME=/path/to/torch/cache

# Debugging
export DEBUG=true
export PYTHONUNBUFFERED=1
```

## Getting Help

1. **Check logs:**
```bash
cat logs/app.log
```

2. **Run system info:**
```bash
python cli.py info
```

3. **Test model loading:**
```python
from diffusers import QwenImageEditPipeline
model = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("Model loaded successfully!")
```

4. **Report issues:** https://github.com/arturwyroslak/qwenedit/issues

## Next Steps

1. Read [README.md](README.md) for usage guide
2. Check [example notebooks](docs/examples.md) for code samples
3. Join community discussions
4. Contribute improvements!
