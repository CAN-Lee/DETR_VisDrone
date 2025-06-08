#!/usr/bin/env python3
"""
Convert VisDrone dataset annotations to COCO format.
"""

import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# VisDrone class mapping
VISDRONE_CLASSES = {
    0: 'ignored_region',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others'
}

def create_coco_annotation(annotation_id, image_id, bbox, category_id, area, iscrowd=0):
    """Create a COCO annotation entry."""
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,  # [x, y, width, height]
        "area": area,
        "iscrowd": iscrowd
    }

def create_coco_image(image_id, file_name, width, height):
    """Create a COCO image entry."""
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def create_coco_category(category_id, name, supercategory="object"):
    """Create a COCO category entry."""
    return {
        "id": category_id,
        "name": name,
        "supercategory": supercategory
    }

def convert_visdrone_to_coco(data_dir, split_name, output_file):
    """
    Convert VisDrone annotations to COCO format.
    
    Args:
        data_dir: Path to VisDrone dataset directory
        split_name: 'train' or 'val'
        output_file: Output JSON file path
    """
    
    # Paths
    images_dir = os.path.join(data_dir, f'VisDrone2019-DET-{split_name}', 'images')
    annotations_dir = os.path.join(data_dir, f'VisDrone2019-DET-{split_name}', 'annotations')
    
    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": f"VisDrone2019-DET-{split_name} dataset in COCO format",
            "url": "http://aiskyeye.com/",
            "version": "1.0",
            "year": 2019,
            "contributor": "VisDrone",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories (skip ignored_region class 0)
    for class_id, class_name in VISDRONE_CLASSES.items():
        if class_id == 0:  # Skip ignored regions
            continue
        coco_output["categories"].append(
            create_coco_category(class_id, class_name)
        )
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort()
    
    annotation_id = 1
    
    print(f"Converting {split_name} split...")
    for image_id, image_file in enumerate(tqdm(image_files), 1):
        # Get image path and annotation path
        image_path = os.path.join(images_dir, image_file)
        annotation_file = image_file.replace('.jpg', '.txt')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue
        
        # Add image info
        coco_output["images"].append(
            create_coco_image(image_id, image_file, width, height)
        )
        
        # Process annotations if file exists
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse VisDrone annotation
                    parts = line.split(',')
                    if len(parts) != 8:
                        continue
                    
                    bbox_left = int(parts[0])
                    bbox_top = int(parts[1])
                    bbox_width = int(parts[2])
                    bbox_height = int(parts[3])
                    score = int(parts[4])  # 1: consider, 0: ignore
                    object_category = int(parts[5])
                    truncation = int(parts[6])
                    occlusion = int(parts[7])
                    
                    # Skip ignored regions (class 0) and ignored annotations (score 0)
                    if object_category == 0 or score == 0:
                        continue
                    
                    # Skip invalid bounding boxes
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue
                    
                    # Calculate area
                    area = bbox_width * bbox_height
                    
                    # Create COCO annotation
                    coco_annotation = create_coco_annotation(
                        annotation_id=annotation_id,
                        image_id=image_id,
                        bbox=[bbox_left, bbox_top, bbox_width, bbox_height],
                        category_id=object_category,
                        area=area,
                        iscrowd=0
                    )
                    
                    coco_output["annotations"].append(coco_annotation)
                    annotation_id += 1
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"Conversion complete!")
    print(f"Output saved to: {output_file}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total annotations: {len(coco_output['annotations'])}")
    print(f"Categories: {len(coco_output['categories'])}")

def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone dataset to COCO format')
    parser.add_argument('--data_dir', default='.',
                        help='Path to VisDrone dataset directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        choices=['train', 'val', 'test-dev', 'test-challenge'],
                        help='Dataset splits to convert')
    parser.add_argument('--output_dir', default='.',
                        help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert each split
    for split in args.splits:
        output_file = os.path.join(args.output_dir, f'visdrone_{split}_coco.json')
        convert_visdrone_to_coco(args.data_dir, split, output_file)

if __name__ == '__main__':
    main() 