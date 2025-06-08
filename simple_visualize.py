#!/usr/bin/env python3
"""
Simple DETR visualization script - generates one detection result image
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Import DETR modules
import datasets
import util.misc as utils
from datasets import build_dataset
from models import build_model
import argparse

# VisDrone class names
VISDRONE_CLASSES = [
    'ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Colors for each class
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471'
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale_tensor = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    b = b * scale_tensor
    return b

def create_args():
    """Create default arguments for the model"""
    class Args:
        def __init__(self):
            # Model parameters
            self.backbone = 'resnet50'
            self.dilation = False
            self.position_embedding = 'sine'
            self.enc_layers = 6
            self.dec_layers = 6
            self.dim_feedforward = 2048
            self.hidden_dim = 256
            self.dropout = 0.1
            self.nheads = 8
            self.num_queries = 100
            self.pre_norm = False
            
            # Loss parameters
            self.aux_loss = True
            self.set_cost_class = 1
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.mask_loss_coef = 1
            self.dice_loss_coef = 1
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.eos_coef = 0.1
            
            # Dataset parameters
            self.dataset_file = 'coco'
            self.coco_path = './VisDrone/VisDrone_COCO'
            self.remove_difficult = False
            self.masks = False
            
            # Training parameters
            self.lr_backbone = 1e-5
            self.lr = 1e-4
            self.lr_drop = 200
            self.weight_decay = 1e-4
            self.clip_max_norm = 0.1
            self.num_classes = 11
            
            # Device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return Args()

def main():
    # Setup
    args = create_args()
    device = torch.device(args.device)
    
    # Build model
    print("Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint_path = './outputs/visdrone_detr_300ep/checkpoint.pth'
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Build dataset
    print("Loading dataset...")
    dataset = build_dataset(image_set='val', args=args)
    print(f"Dataset has {len(dataset)} images")
    
    # Select the first image
    img, target = dataset[0]
    image_id = target['image_id'].item()
    
    # Load original image
    img_path = dataset.coco.loadImgs(image_id)[0]['file_name']
    image_path = Path(args.coco_path) / 'val2017' / img_path
    original_image = Image.open(image_path).convert('RGB')
    print(f"Processing image: {img_path}")
    
    # Prepare input
    input_tensor = img.unsqueeze(0).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Process predictions
    threshold = 0.3
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Remove no-object class
    keep = probas.max(-1).values > threshold
    
    if keep.sum() == 0:
        print("No detections above threshold!")
        return
    
    # Convert boxes to image coordinates
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], original_image.size)
    probas = probas[keep]
    
    print(f"Found {len(bboxes_scaled)} detections above threshold {threshold}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(original_image)
    ax.set_title(f'DETR Detection Results on VisDrone (threshold={threshold})', fontsize=16)
    
    # Draw bounding boxes
    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
        cl = p.argmax()
        confidence = p.max().item()
        
        # Draw rectangle
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               linewidth=3, edgecolor=COLORS[cl % len(COLORS)], 
                               facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = f'{VISDRONE_CLASSES[cl]}: {confidence:.2f}'
        ax.text(xmin, ymin - 8, label, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS[cl % len(COLORS)], alpha=0.7),
                color='white', weight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save the result
    output_path = 'detection_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_path}")
    
    plt.close()

if __name__ == '__main__':
    main() 