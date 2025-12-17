#!/usr/bin/env python3
"""
Configuration module for Qwen Image Edit application.
"""

import os
from pathlib import Path
from typing import Optional
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit"

# Device selection
def get_device() -> str:
    """
    Select the best available device.
    Priority: CUDA GPU > MPS (Mac) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Precision configuration
def get_precision(device: str = None) -> torch.dtype:
    """
    Select optimal precision based on device.
    
    Args:
        device: Device type ('cuda', 'mps', 'cpu')
        
    Returns:
        PyTorch dtype
    """
    if device is None:
        device = get_device()
    
    if device == "cuda":
        # CUDA: Use BF16 for better performance
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device == "mps":
        # Metal Performance Shaders: Use BF16
        return torch.bfloat16
    else:
        # CPU: Use BF16 (PyTorch CPU BF16 is efficient)
        return torch.bfloat16

# Inference settings
INFERENCE_SETTINGS = {
    "cpu": {
        "dtype": torch.bfloat16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "enable_attention_slicing": True,
        "enable_sequential_cpu_offload": False,  # CPU doesn't benefit from this
        "max_image_size": 1024,
    },
    "cuda": {
        "dtype": torch.bfloat16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "enable_attention_slicing": False,
        "enable_sequential_cpu_offload": True,
        "max_image_size": 1024,
    },
    "mps": {
        "dtype": torch.bfloat16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "enable_attention_slicing": True,
        "enable_sequential_cpu_offload": False,
        "max_image_size": 768,
    },
}

# Gradio settings
GRADIO_SETTINGS = {
    "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    "share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
    "debug": os.getenv("DEBUG", "false").lower() == "true",
}

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# Prompts for different tasks
HAIRSTYLE_PROMPTS = {
    "transfer": (
        "Transfer the exact hairstyle, hair texture, hair color, and hair styling from image1 "
        "onto the person in image2. Keep the person's face and body in image2 intact, "
        "only modify the hair to match image1's hairstyle perfectly. "
        "Maintain natural lighting and overall image quality."
    ),
    "enhance": (
        "Enhance the hairstyle in the image. Make the hair look more polished, shinier, "
        "and well-maintained while preserving the original style and color."
    ),
    "color_change": (
        "Keep the hairstyle and texture from image1, but change the hair color to match image2's hair color. "
        "Maintain natural lighting and avoid any other modifications."
    ),
}

# Example prompts for reference
EXAMPLE_PROMPTS = {
    "basic": "Transfer the exact hairstyle from image1 onto the person in image2. Keep everything else the same.",
    "detailed": "Transfer the wavy, long hairstyle and brunette color from image1 onto the woman in image2. Keep her face and body intact.",
    "style_only": "Copy the curly perm hairstyle from image1 to image2, keeping the original hair color.",
    "color_style": "Transfer both the blonde color and the bob cut hairstyle from image1 to the person in image2.",
}

if __name__ == "__main__":
    print(f"Device: {get_device()}")
    print(f"Precision: {get_precision()}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")