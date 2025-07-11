#!/usr/bin/env python3
"""
LaserWeed Batch Model Conversion Script

This script automatically detects and converts all .ckpt files in the custom-models
directory to TensorFlow.js format for the LaserWeed web interface.

Usage:
    python batch_convert.py
    python batch_convert.py --quantize
    python batch_convert.py --models-dir /path/to/models
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Import the main converter
from convert_model import ModelConverter, validate_model

class BatchConverter:
    """Convert multiple YOLO models in batch."""
    
    def __init__(self, models_dir: str = "custom-models", quantize: bool = False):
        self.models_dir = Path(models_dir)
        self.quantize = quantize
        self.results = []
        
    def scan_for_models(self) -> List[Tuple[Path, str]]:
        """Scan directory for convertible model files."""
        models = []
        
        if not self.models_dir.exists():
            print(f"Models directory not found: {self.models_dir}")
            return models
        
        # Look for .ckpt, .pt, and .onnx files
        for extension in ['.ckpt', '.pt', '.onnx']:
            for model_file in self.models_dir.rglob(f'*{extension}'):
                # Skip if already converted (check for corresponding model.json)
                output_dir = model_file.parent / model_file.stem
                if (output_dir / "model.json").exists():
                    print(f"⏭️  Skipping {model_file.name} - already converted")
                    continue
                
                models.append((model_file, self.determine_model_type(model_file)))
        
        return models
    
    def determine_model_type(self, model_path: Path) -> str:
        """Determine model type from filename."""
        name = model_path.name.lower()
        
        if 'yolov7' in name or model_path.suffix == '.ckpt':
            if 'tiny' in name:
                return 'yolov7-tiny'
            elif 'x' in name:
                return 'yolov7-x'
            elif 'e6' in name:
                return 'yolov7-e6'
            else:
                return 'yolov7'
        elif 'yolov5' in name:
            if 's' in name:
                return 'yolov5s'
            elif 'm' in name:
                return 'yolov5m'
            elif 'l' in name:
                return 'yolov5l'
            else:
                return 'yolov5'
        elif 'yolov8' in name:
            return 'yolov8'
        else:
            return 'unknown'
    
    def convert_model(self, model_path: Path, model_type: str) -> Dict:
        """Convert a single model."""
        print(f"\n🔄 Converting {model_path.name} ({model_type})...")
        
        # Create output directory
        output_dir = model_path.parent / model_path.stem
        
        try:
            start_time = time.time()
            
            # Convert model
            converter = ModelConverter(str(model_path), str(output_dir), self.quantize)
            success = converter.convert()
            
            conversion_time = time.time() - start_time
            
            if success:
                # Validate converted model
                is_valid = validate_model(output_dir)
                
                if is_valid:
                    # Get model size
                    model_size = self.calculate_model_size(output_dir)
                    
                    result = {
                        'model': model_path.name,
                        'type': model_type,
                        'status': 'success',
                        'output_dir': str(output_dir),
                        'size_mb': model_size,
                        'conversion_time': round(conversion_time, 2),
                        'quantized': self.quantize
                    }
                    
                    print(f"✅ {model_path.name} converted successfully ({model_size:.1f}MB, {conversion_time:.1f}s)")
                else:
                    result = {
                        'model': model_path.name,
                        'type': model_type,
                        'status': 'validation_failed',
                        'error': 'Model validation failed'
                    }
                    print(f"❌ {model_path.name} validation failed")
            else:
                result = {
                    'model': model_path.name,
                    'type': model_type,
                    'status': 'conversion_failed',
                    'error': 'Model conversion failed'
                }
                print(f"❌ {model_path.name} conversion failed")
                
        except Exception as e:
            result = {
                'model': model_path.name,
                'type': model_type,
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ {model_path.name} error: {e}")
        
        return result
    
    def calculate_model_size(self, model_dir: Path) -> float:
        """Calculate total size of converted model files in MB."""
        total_size = 0
        
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def convert_all(self) -> None:
        """Convert all detected models."""
        models = self.scan_for_models()
        
        if not models:
            print("🔍 No models found for conversion.")
            print(f"Place .ckpt, .pt, or .onnx files in: {self.models_dir}")
            return
        
        print(f"🚀 Found {len(models)} models to convert:")
        for model_path, model_type in models:
            print(f"  • {model_path.name} ({model_type})")
        
        print(f"\n⚙️  Quantization: {'Enabled (FP16)' if self.quantize else 'Disabled'}")
        
        # Convert each model
        total_start_time = time.time()
        
        for model_path, model_type in models:
            result = self.convert_model(model_path, model_type)
            self.results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Print summary
        self.print_summary(total_time)
        
        # Save results
        self.save_results()
    
    def print_summary(self, total_time: float) -> None:
        """Print conversion summary."""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] != 'success']
        
        print(f"\n📊 Conversion Summary:")
        print(f"   Total models: {len(self.results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        print(f"   Total time: {total_time:.1f}s")
        
        if successful:
            total_size = sum(r.get('size_mb', 0) for r in successful)
            avg_time = sum(r.get('conversion_time', 0) for r in successful) / len(successful)
            
            print(f"\n✅ Successfully converted models:")
            for result in successful:
                print(f"   • {result['model']} ({result['type']}) - {result['size_mb']:.1f}MB")
            
            print(f"\n📈 Performance:")
            print(f"   Total size: {total_size:.1f}MB")
            print(f"   Average conversion time: {avg_time:.1f}s")
            print(f"   Quantization: {'FP16' if self.quantize else 'FP32'}")
        
        if failed:
            print(f"\n❌ Failed conversions:")
            for result in failed:
                print(f"   • {result['model']} - {result.get('error', 'Unknown error')}")
    
    def save_results(self) -> None:
        """Save conversion results to JSON file."""
        results_file = self.models_dir / "conversion_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'quantized': self.quantize,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def generate_model_index(self) -> None:
        """Generate an index of available models for the web interface."""
        successful_models = [r for r in self.results if r['status'] == 'success']
        
        if not successful_models:
            return
        
        index = {
            'models': [],
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': len(successful_models)
        }
        
        for result in successful_models:
            model_info = {
                'id': Path(result['model']).stem,
                'name': result['model'],
                'type': result['type'],
                'path': f"models/{Path(result['output_dir']).name}/model.json",
                'config_path': f"models/{Path(result['output_dir']).name}/classes.json",
                'size_mb': result['size_mb'],
                'quantized': result['quantized'],
                'performance_tier': self.get_performance_tier(result['type'])
            }
            index['models'].append(model_info)
        
        # Save index file
        index_file = Path("web/models_index.json")
        index_file.parent.mkdir(exist_ok=True)
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"📋 Model index generated: {index_file}")
    
    def get_performance_tier(self, model_type: str) -> str:
        """Get performance tier for model type."""
        if 'tiny' in model_type:
            return 'fast'
        elif 'x' in model_type or 'l' in model_type:
            return 'accurate'
        elif 'e6' in model_type:
            return 'premium'
        else:
            return 'balanced'


def main():
    parser = argparse.ArgumentParser(description='Batch convert YOLO models to TensorFlow.js')
    parser.add_argument('--models-dir', '-d', default='custom-models', 
                       help='Directory containing model files (default: custom-models)')
    parser.add_argument('--quantize', '-q', action='store_true', 
                       help='Apply FP16 quantization for smaller models')
    parser.add_argument('--generate-index', '-i', action='store_true',
                       help='Generate model index for web interface')
    
    args = parser.parse_args()
    
    print("🔥 LaserWeed Batch Model Converter")
    print("=" * 40)
    
    # Create batch converter
    converter = BatchConverter(args.models_dir, args.quantize)
    
    # Convert all models
    converter.convert_all()
    
    # Generate index if requested
    if args.generate_index:
        converter.generate_model_index()
    
    print("\n🎯 Ready for LaserWeed!")
    print("Copy the converted models to your web/models/ directory.")


if __name__ == "__main__":
    main()