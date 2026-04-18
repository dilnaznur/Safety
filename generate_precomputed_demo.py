#!/usr/bin/env python
"""
Generate precomputed detection results for demo samples
This pre-processes demo videos offline for instant playback during presentations

Usage:
    python generate_precomputed_demo.py
    python generate_precomputed_demo.py --video path/to/video.mp4
    python generate_precomputed_demo.py --all
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from demo_mode import DemoPrecomputeGenerator
from app import engine, logger


def main():
    parser = argparse.ArgumentParser(
        description="Generate precomputed detection results for demo samples"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Process specific video file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all videos in samples/ directory"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="samples",
        help="Directory containing sample videos (default: samples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/precomputed",
        help="Output directory for precomputed results (default: samples/precomputed)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing precomputed results"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create generator
    generator = DemoPrecomputeGenerator(args.samples_dir, args.output_dir)
    
    print("\n" + "="*70)
    print("SafetyVision AI - Demo Precomputation Generator")
    print("="*70)
    print(f"Sample directory: {args.samples_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")
    
    try:
        if args.video:
            # Process specific video
            if not os.path.exists(args.video):
                print(f"❌ Video file not found: {args.video}")
                return 1
            
            output_file = os.path.join(
                args.output_dir,
                Path(args.video).stem + ".json"
            )
            
            if os.path.exists(output_file) and not args.force:
                print(f"⚠️  Output already exists: {output_file}")
                print("   Use --force to overwrite")
                return 0
            
            print(f"Processing: {args.video}")
            success = generator.generate_for_sample(args.video, engine)
            
            if success:
                print(f"✅ Successfully generated: {output_file}\n")
                return 0
            else:
                print(f"❌ Failed to process: {args.video}\n")
                return 1
        
        elif args.all:
            # Process all videos
            print("Processing all videos in samples/...\n")
            generator.generate_all(engine)
            print("\n✅ All videos processed!\n")
            return 0
        
        else:
            # Interactive mode
            samples_dir = args.samples_dir
            if not os.path.exists(samples_dir):
                print(f"❌ Samples directory not found: {samples_dir}")
                print("   Create it and add .mp4/.avi videos")
                return 1
            
            video_files = [
                f for f in os.listdir(samples_dir)
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            ]
            
            if not video_files:
                print(f"❌ No video files found in: {samples_dir}")
                return 1
            
            print(f"Found {len(video_files)} sample(s):\n")
            for i, vf in enumerate(video_files, 1):
                video_path = os.path.join(samples_dir, vf)
                output_file = os.path.join(args.output_dir, Path(vf).stem + ".json")
                status = "✅ (cached)" if os.path.exists(output_file) else "⏳ (new)"
                print(f"  {i}. {vf} {status}")
            
            print("\nProcessing all samples...\n")
            generator.generate_all(engine)
            
            print("\n" + "="*70)
            print("✅ Precomputation Complete!")
            print("="*70)
            print(f"\nYour demo is now ready for instant playback!")
            print(f"Start demo mode with:")
            print(f"  export DEPLOYMENT_MODE=demo")
            print(f"  export TARGET_FPS=15")
            print(f"  uvicorn app:app --host 0.0.0.0 --port 8000")
            print(f"\nThen open: http://localhost:8000")
            print("="*70 + "\n")
            
            return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Precomputation error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
