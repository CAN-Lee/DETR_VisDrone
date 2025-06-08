# VisDrone Dataset in COCO Format

## 目录结构

已将VisDrone数据集整理为COCO格式，目录结构如下：

```
VisDrone_COCO/
├── annotations/
│   ├── instances_train2017.json    # 训练集标注 (64MB)
│   ├── instances_val2017.json      # 验证集标注 (7.2MB)
│   └── instances_test2017.json     # 测试集标注 (14MB) - 仅test-dev有标注
├── train2017/                      # 训练图片 (6,471张)
├── val2017/                        # 验证图片 (548张)
└── test2017/                       # 测试图片 (1,610张) - 仅test-dev数据
```

## 数据统计

- **训练图片**: 6,471张 (有标注)
- **验证图片**: 548张 (有标注)
- **测试图片**: 1,610张 (test-dev, 有标注)
- **标注格式**: COCO JSON格式
- **总标注数**: 
  - 训练集: 390,651个目标
  - 验证集: 33,910个目标
  - 测试集: 75,102个目标

## 在DETR中使用

### 1. 训练命令

```bash
# 单卡训练
python main.py --coco_path /data/lican/vision_transformer_course/3/detr/VisDrone/VisDrone_COCO

# 多卡训练 (8卡)
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /data/lican/vision_transformer_course/3/detr/VisDrone/VisDrone_COCO
```

### 2. 评估命令

```bash
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/checkpoint.pth --coco_path /data/lican/vision_transformer_course/3/detr/VisDrone/VisDrone_COCO
```

### 3. 测试推理

```bash
# 在测试集上进行推理 (无标注评估)
python main.py --batch_size 2 --no_aux_loss --eval --resume /path/to/checkpoint.pth --coco_path /data/lican/vision_transformer_course/3/detr/VisDrone/VisDrone_COCO --test
```

### 4. 绝对路径

训练时使用的完整路径：
```
/data/lican/vision_transformer_course/3/detr/VisDrone/VisDrone_COCO
```

## 注意事项

1. 标注文件已从原始格式转换为COCO标准格式
2. 图片文件已从原始目录复制到 `train2017/`、`val2017/` 和 `test2017/` 目录
3. **测试集更新**: 目前仅包含test-dev数据(1,610张)，因为只有test-dev有标注文件
4. test-challenge数据无标注文件，暂未包含在内
5. 目录结构完全符合DETR训练要求
6. 可以直接使用 `--coco_path` 参数指定 `VisDrone_COCO` 目录路径

## 原始数据

原始VisDrone数据集仍保留在以下目录中：
- `VisDrone2019-DET-train/`
- `VisDrone2019-DET-val/`
- `VisDrone2019-DET-test-dev/`
- `VisDrone2019-DET-test-challenge/` 