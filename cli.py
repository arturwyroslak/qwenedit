#!/usr/bin/env python3
"""
Command-line interface for Qwen Image Edit hairstyle transfer.
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
import torch
from app import HairstyleTransferApp
from config import get_device, get_precision, OUTPUT_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def transfer_command(args):
    """
    Handle the transfer command.
    """
    try:
        # Validate input files
        source_path = Path(args.source)
        target_path = Path(args.target)
        
        if not source_path.exists():
            logger.error(f"Source image not found: {source_path}")
            return 1
        
        if not target_path.exists():
            logger.error(f"Target image not found: {target_path}")
            return 1
        
        # Load images
        logger.info(f"Loading source image: {source_path}")
        source_image = Image.open(source_path).convert('RGB')
        
        logger.info(f"Loading target image: {target_path}")
        target_image = Image.open(target_path).convert('RGB')
        
        # Initialize app
        device = args.device or get_device()
        logger.info(f"Using device: {device}")
        
        app = HairstyleTransferApp(
            device=device,
            precision=get_precision(device)
        )
        
        # Run transfer
        logger.info("Starting hairstyle transfer...")
        result_image, status = app.transfer_hairstyle(
            source_image,
            target_image,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            negative_prompt=args.negative_prompt,
            seed=args.seed
        )
        
        if result_image is None:
            logger.error("Transfer failed")
            return 1
        
        # Save output
        output_path = Path(args.output) if args.output else OUTPUT_DIR / "output.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving output image: {output_path}")
        result_image.save(output_path)
        
        logger.info(f"Transfer complete: {status}")
        print(f"✅ Output saved to: {output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def batch_command(args):
    """
    Handle the batch command for multiple variations.
    """
    try:
        # Validate input files
        source_path = Path(args.source)
        target_path = Path(args.target)
        
        if not source_path.exists():
            logger.error(f"Source image not found: {source_path}")
            return 1
        
        if not target_path.exists():
            logger.error(f"Target image not found: {target_path}")
            return 1
        
        # Load images
        logger.info(f"Loading source image: {source_path}")
        source_image = Image.open(source_path).convert('RGB')
        
        logger.info(f"Loading target image: {target_path}")
        target_image = Image.open(target_path).convert('RGB')
        
        # Initialize app
        device = args.device or get_device()
        logger.info(f"Using device: {device}")
        
        app = HairstyleTransferApp(
            device=device,
            precision=get_precision(device)
        )
        
        # Run batch transfer
        logger.info(f"Generating {args.count} variations...")
        results, status = app.batch_transfer(
            source_image,
            target_image,
            num_variations=args.count
        )
        
        if not results:
            logger.error("Batch generation failed")
            return 1
        
        # Save outputs
        output_dir = Path(args.output) if args.output else OUTPUT_DIR / "batch"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img in enumerate(results):
            output_path = output_dir / f"variation_{idx+1}.png"
            logger.info(f"Saving variation {idx+1}: {output_path}")
            img.save(output_path)
        
        logger.info(f"Batch generation complete: {status}")
        print(f"✅ Generated {len(results)} variations in: {output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def info_command(args):
    """
    Display system information.
    """
    print("\n🖥️  System Information:")
    print(f"  Device: {get_device()}")
    print(f"  Precision: {get_precision()}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  PyTorch Version: {torch.__version__}")
    print()
    return 0


def main():
    """
    Main entry point for CLI.
    """
    parser = argparse.ArgumentParser(
        description="Qwen Image Edit - Hairstyle Transfer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single transfer
  python cli.py transfer source.jpg target.jpg -o output.jpg
  
  # Transfer with custom parameters
  python cli.py transfer source.jpg target.jpg -o output.jpg --steps 50 --cfg 5.0
  
  # Generate 3 variations
  python cli.py batch source.jpg target.jpg -o output_dir/ --count 3
  
  # Show system info
  python cli.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Transfer command
    transfer_parser = subparsers.add_parser("transfer", help="Transfer hairstyle between images")
    transfer_parser.add_argument("source", help="Source image (hairstyle donor)")
    transfer_parser.add_argument("target", help="Target image (person to receive hairstyle)")
    transfer_parser.add_argument("-o", "--output", help="Output image path")
    transfer_parser.add_argument("--steps", type=int, default=40, help="Number of inference steps (default: 40)")
    transfer_parser.add_argument("--cfg", type=float, default=4.0, help="Guidance scale (default: 4.0)")
    transfer_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    transfer_parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Device to use (auto-detect by default)")
    transfer_parser.add_argument(
        "--negative-prompt",
        default="ugly, blurry, distorted, deformed, low quality",
        help="Negative prompt"
    )
    transfer_parser.set_defaults(func=transfer_command)
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Generate multiple variations")
    batch_parser.add_argument("source", help="Source image (hairstyle donor)")
    batch_parser.add_argument("target", help="Target image (person to receive hairstyle)")
    batch_parser.add_argument("-o", "--output", help="Output directory")
    batch_parser.add_argument("--count", type=int, default=3, help="Number of variations (default: 3)")
    batch_parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Device to use (auto-detect by default)")
    batch_parser.set_defaults(func=batch_command)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    info_parser.set_defaults(func=info_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    exit(main())