#!/usr/bin/env python3
"""
Visualize DETR detection results on VisDrone test images
"""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

import datasets
import util.misc as utils
from datasets import build_dataset
from models import build_model

# VisDrone class names
VISDRONE_CLASSES = [
    'ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Colors for visualization
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471'
]

def get_args_parser():
    parser = argparse.ArgumentParser('DETR visualization', add_help=False)
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    
    # Loss parameters (not used in inference)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    
    # Dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--masks', action='store_true')
    
    # Inference parameters
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--resume', required=True, help='checkpoint path')
    parser.add_argument('--threshold', default=0.5, type=float, help='confidence threshold')
    parser.add_argument('--num_images', default=10, type=int, help='number of images to visualize')
    parser.add_argument('--output_dir', default='./visualization_results', help='output directory')
    
    # Additional required parameters
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--num_classes', default=11, type=int, help='number of classes (VisDrone has 11 classes)')
    
    return parser

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def visualize_predictions(image, predictions, threshold=0.5, save_path=None):
    """
    Visualize detection results on image
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f'DETR Detection Results (threshold={threshold})')
    
    # Get predictions
    probas = predictions['pred_logits'].softmax(-1)[0, :, :-1]  # Remove no-object class
    keep = probas.max(-1).values > threshold
    
    # Convert boxes to image coordinates
    bboxes_scaled = rescale_bboxes(predictions['pred_boxes'][0, keep], image.size)
    probas = probas[keep]
    
    # Draw bounding boxes
    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
        cl = p.argmax()
        confidence = p.max().item()
        
        # Draw rectangle
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        linewidth=2, edgecolor=COLORS[cl % len(COLORS)], 
                        facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add text
        label = f'{VISDRONE_CLASSES[cl]}: {confidence:.2f}'
        ax.text(xmin, ymin - 5, label, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS[cl % len(COLORS)], alpha=0.7),
                color='white', weight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    return fig

def main(args):
    device = torch.device(args.device)
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Build dataset (using validation set for visualization)
    dataset = build_dataset(image_set='val', args=args)
    print(f"Dataset has {len(dataset)} images")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transform for inference
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Select random images for visualization
    indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))
    
    print(f"Visualizing {len(indices)} images...")
    
    for i, idx in enumerate(indices):
        print(f"Processing image {i+1}/{len(indices)} (index {idx})")
        
        # Load image and target
        img, target = dataset[idx]
        image_id = target['image_id'].item()
        
        # Load original image for visualization
        img_path = dataset.coco.loadImgs(image_id)[0]['file_name']
        image_path = Path(args.coco_path) / 'val2017' / img_path
        original_image = Image.open(image_path).convert('RGB')
        
        # Prepare input
        input_tensor = img.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Post-process outputs
        orig_target_sizes = torch.tensor([original_image.size[::-1]], device=device)  # (height, width)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # Visualize and save
        save_path = output_dir / f'detection_result_{i:03d}_id{image_id}.png'
        # Use raw outputs for visualization
        visualize_predictions(original_image, outputs, threshold=args.threshold, save_path=save_path)
        
        # Also save original image for comparison
        original_image.save(output_dir / f'original_{i:03d}_id{image_id}.png')
    
    print(f"Visualization complete! Results saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR visualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 