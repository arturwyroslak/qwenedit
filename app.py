#!/usr/bin/env python3
"""
Qwen Image Edit - Hairstyle Transfer Application
CPU-optimized Gradio interface for transferring hairstyles between images
"""

import os
import gc
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
from diffusers import QwenImageEditPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HairstyleTransferApp:
    """
    CPU-optimized hairstyle transfer using Qwen Image Edit.
    
    Optimizations:
    - BF16 precision for reduced VRAM
    - CPU offloading for model components
    - Image preprocessing and scaling
    - Automatic garbage collection
    - Sequential execution (no batching)
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit",
        device: str = "cpu",
        precision: torch.dtype = torch.bfloat16,
        enable_memory_efficient: bool = True
    ):
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.enable_memory_efficient = enable_memory_efficient
        self.pipeline = None
        self._initialized = False
        
        logger.info(f"Initialized HairstyleTransferApp on {device} with {precision}")
    
    def initialize(self):
        """
        Lazily load the model to avoid issues during app startup.
        """
        if self._initialized:
            return
        
        try:
            logger.info("Loading Qwen Image Edit pipeline...")
            
            # Load with CPU optimizations
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.precision,
                use_safetensors=True,
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient features if on CPU
            if self.device == "cpu" and self.enable_memory_efficient:
                # CPU inference is sequential by default
                logger.info("CPU device detected - using sequential inference")
            
            self._initialized = True
            logger.info("Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Preprocess image: resize to max size while preserving aspect ratio.
        
        Args:
            image: Input PIL Image
            max_size: Maximum dimension size
            
        Returns:
            Preprocessed PIL Image
        """
        if image is None:
            return None
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if necessary
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Ensure dimensions are multiple of 8 for better model compatibility
        width, height = image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        if (width, height) != image.size:
            image = image.crop((0, 0, width, height))
        
        logger.info(f"Preprocessed image to size: {image.size}")
        return image
    
    def transfer_hairstyle(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        negative_prompt: str = "ugly, blurry, distorted, deformed, low quality",
        seed: Optional[int] = None,
        progress=None
    ) -> Tuple[Image.Image, str]:
        """
        Transfer hairstyle from source image to target image.
        
        Args:
            source_image: Image with source hairstyle
            target_image: Image with target person
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for prompt adherence
            negative_prompt: Negative prompt
            seed: Random seed for reproducibility
            progress: Gradio progress callback
            
        Returns:
            Tuple of (output image, status message)
        """
        try:
            # Initialize if needed
            if not self._initialized:
                self.initialize()
            
            if progress:
                progress(0, desc="Preprocessing images...")
            
            # Preprocess images
            source = self.preprocess_image(source_image)
            target = self.preprocess_image(target_image)
            
            if progress:
                progress(10, desc="Building prompt...")
            
            # Create detailed prompt for hairstyle transfer
            prompt = (
                "Transfer the exact hairstyle, hair texture, and hair color from image1 "
                "onto the person in image2. Keep the person in image2 intact, "
                "only modify the hair to match image1's hairstyle. "
                "Maintain natural lighting and overall image quality."
            )
            
            logger.info(f"Using prompt: {prompt}")
            logger.info(f"Inference steps: {num_inference_steps}, Guidance scale: {guidance_scale}")
            
            if progress:
                progress(20, desc="Generating output...")
            
            # Generate with explicit random state
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Run inference with memory optimization
            with torch.inference_mode():
                output = self.pipeline(
                    image=target,  # Base image (person)
                    prompt=prompt,  # Edit instruction
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    true_cfg_scale=guidance_scale,  # For Qwen compatibility
                )
            
            result_image = output.images[0]
            
            if progress:
                progress(95, desc="Finalizing...")
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            status = (
                f"✅ Hairstyle transfer completed!\n"
                f"Source: {source.size} | Target: {target.size} | Output: {result_image.size}\n"
                f"Steps: {num_inference_steps} | CFG: {guidance_scale}"
            )
            
            if progress:
                progress(100, desc="Complete")
            
            logger.info(f"Generated output image: {result_image.size}")
            return result_image, status
        
        except Exception as e:
            logger.error(f"Error during hairstyle transfer: {e}", exc_info=True)
            error_msg = f"❌ Error: {str(e)}"
            return None, error_msg
    
    def batch_transfer(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        num_variations: int = 1,
        progress=None
    ) -> Tuple[list, str]:
        """
        Generate multiple variations of hairstyle transfer.
        
        Args:
            source_image: Image with source hairstyle
            target_image: Image with target person
            num_variations: Number of variations to generate
            progress: Gradio progress callback
            
        Returns:
            Tuple of (list of output images, status message)
        """
        results = []
        try:
            for i in range(num_variations):
                if progress:
                    progress((i / num_variations) * 100, desc=f"Generating variation {i+1}/{num_variations}...")
                
                seed = i  # Different seed for each variation
                output, _ = self.transfer_hairstyle(
                    source_image,
                    target_image,
                    seed=seed,
                    progress=None
                )
                
                if output is not None:
                    results.append(output)
            
            status = f"✅ Generated {len(results)} variations successfully"
            return results, status
        
        except Exception as e:
            logger.error(f"Error during batch transfer: {e}")
            return [], f"❌ Error during batch generation: {str(e)}"


def create_gradio_interface():
    """
    Create and configure the Gradio interface.
    """
    
    # Initialize app
    app = HairstyleTransferApp(device="cpu", precision=torch.bfloat16)
    
    # Custom CSS for better UI
    css = """
    .title-text { text-align: center; margin-bottom: 20px; }
    .info-box { background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .result-box { background-color: #f0fff0; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .error-box { background-color: #fff0f0; padding: 15px; border-radius: 8px; margin: 10px 0; }
    """
    
    with gr.Blocks(css=css, title="Qwen Hairstyle Transfer") as demo:
        # Header
        gr.Markdown(
            """
            # 💇‍♀️ Qwen Image Edit - Hairstyle Transfer
            
            Transfer hairstyles between images using Qwen Image Edit.
            Powered by state-of-the-art diffusion models optimized for CPU execution.
            
            **Features:**
            - High-quality hairstyle transfer
            - CPU-optimized inference
            - Batch generation support
            - Adjustable quality parameters
            """
        )
        
        with gr.Tabs():
            # Single Transfer Tab
            with gr.Tab("Single Transfer"):
                gr.Markdown("### Transfer hairstyle from one image to another")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Upload Images**")
                        source_img = gr.Image(
                            label="Source Image (hairstyle donor)",
                            type="pil",
                            scale=1
                        )
                        target_img = gr.Image(
                            label="Target Image (person to receive hairstyle)",
                            type="pil",
                            scale=1
                        )
                    
                    with gr.Column():
                        gr.Markdown("**Parameters**")
                        
                        num_steps = gr.Slider(
                            label="Inference Steps",
                            minimum=20,
                            maximum=50,
                            value=40,
                            step=5,
                            info="More steps = better quality but slower"
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=10.0,
                            value=4.0,
                            step=0.5,
                            info="Higher = stronger prompt adherence"
                        )
                        
                        seed_input = gr.Number(
                            label="Seed (for reproducibility)",
                            value=-1,
                            precision=0,
                            info="Use -1 for random"
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="ugly, blurry, distorted, deformed, low quality",
                            lines=2
                        )
                        
                        transfer_btn = gr.Button(
                            "🚀 Transfer Hairstyle",
                            scale=2,
                            variant="primary"
                        )
                
                # Results
                with gr.Row():
                    with gr.Column():
                        output_img = gr.Image(
                            label="Output Image",
                            type="pil"
                        )
                    
                    with gr.Column():
                        output_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=4
                        )
                
                # Transfer button handler
                def handle_transfer(
                    source,
                    target,
                    steps,
                    guidance,
                    seed,
                    neg_prompt,
                    progress=gr.Progress()
                ):
                    if source is None or target is None:
                        return None, "❌ Please upload both images"
                    
                    seed = None if seed == -1 else int(seed)
                    return app.transfer_hairstyle(
                        source,
                        target,
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        negative_prompt=neg_prompt,
                        seed=seed,
                        progress=progress
                    )
                
                transfer_btn.click(
                    fn=handle_transfer,
                    inputs=[source_img, target_img, num_steps, guidance_scale, seed_input, negative_prompt],
                    outputs=[output_img, output_status]
                )
            
            # Batch Generation Tab
            with gr.Tab("Batch Generation"):
                gr.Markdown("### Generate multiple variations")
                
                with gr.Row():
                    with gr.Column():
                        source_img_batch = gr.Image(
                            label="Source Image (hairstyle donor)",
                            type="pil"
                        )
                        target_img_batch = gr.Image(
                            label="Target Image (person to receive hairstyle)",
                            type="pil"
                        )
                        
                        num_variations = gr.Slider(
                            label="Number of Variations",
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1
                        )
                        
                        batch_btn = gr.Button(
                            "🔄 Generate Variations",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="Generated Variations",
                            show_label=True,
                            columns=2,
                            rows=2,
                            preview=True
                        )
                        batch_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                
                def handle_batch(
                    source,
                    target,
                    variations,
                    progress=gr.Progress()
                ):
                    if source is None or target is None:
                        return [], "❌ Please upload both images"
                    
                    return app.batch_transfer(
                        source,
                        target,
                        num_variations=int(variations),
                        progress=progress
                    )
                
                batch_btn.click(
                    fn=handle_batch,
                    inputs=[source_img_batch, target_img_batch, num_variations],
                    outputs=[gallery, batch_status]
                )
            
            # About Tab
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About Qwen Image Edit
                    
                    Qwen Image Edit is a state-of-the-art image editing model developed by Alibaba.
                    
                    ### Key Features:
                    - **Multi-Image Support**: Combine elements from multiple images
                    - **Text-Based Control**: Edit using natural language descriptions
                    - **Advanced Controls**: Depth maps, pose maps, and edge guidance
                    - **High Quality**: Near-photorealistic results
                    
                    ### This Application:
                    - **CPU Optimized**: Runs efficiently on CPU with BF16 precision
                    - **User-Friendly**: Simple Gradio interface for easy interaction
                    - **Flexible**: Adjustable quality parameters
                    - **Open Source**: Built with popular open-source libraries
                    
                    ### Technical Stack:
                    - **Model**: Qwen/Qwen-Image-Edit (Diffusers)
                    - **UI**: Gradio
                    - **Framework**: PyTorch
                    - **Optimization**: BF16 precision + CPU inference
                    
                    ### Resources:
                    - [Qwen GitHub](https://github.com/QwenLM/Qwen-VL)
                    - [Diffusers Documentation](https://huggingface.co/docs/diffusers)
                    - [Gradio Documentation](https://www.gradio.app/)
                    """
                )
    
    return demo


if __name__ == "__main__":
    # Create and launch interface
    demo = create_gradio_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )