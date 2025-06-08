#!/usr/bin/env python3
"""
Simple script to convert VisDrone annotations to COCO format.
"""

import os
import sys
from visdrone_to_coco import convert_visdrone_to_coco

def main():
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir  # VisDrone folder
    
    # Convert train and val splits
    splits = ['train', 'val']
    
    for split in splits:
        print(f"\n{'='*50}")
        print(f"Converting {split} split...")
        print(f"{'='*50}")
        
        output_file = os.path.join(data_dir, f'visdrone_{split}_coco.json')
        
        try:
            convert_visdrone_to_coco(data_dir, split, output_file)
        except Exception as e:
            print(f"Error converting {split} split: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Conversion completed!")
    print(f"{'='*50}")

if __name__ == '__main__':
    main() 