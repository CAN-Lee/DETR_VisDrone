# VisDrone to COCO Format Converter

这个工具可以将VisDrone数据集的标注格式转换为COCO格式的JSON文件。

## 文件说明

- `visdrone_to_coco.py`: 主要的转换脚本，包含所有转换逻辑
- `convert_annotations.py`: 简单的运行脚本，用于快速转换train和val数据
- `README.md`: 使用说明文档

## 数据集结构

确保你的VisDrone数据集结构如下：

```
VisDrone/
├── VisDrone2019-DET-train/
│   ├── images/
│   │   ├── 0000002_00005_d_0000014.jpg
│   │   └── ...
│   └── annotations/
│       ├── 0000002_00005_d_0000014.txt
│       └── ...
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
└── ...
```

## 使用方法

### 方法1: 使用简单脚本 (推荐)

```bash
cd VisDrone
python convert_annotations.py
```

这将自动转换train和val数据集，生成：
- `visdrone_train_coco.json`
- `visdrone_val_coco.json`

### 方法2: 使用主脚本 (更多选项)

```bash
cd VisDrone

# 转换train和val (默认)
python visdrone_to_coco.py --data_dir . --splits train val

# 只转换train
python visdrone_to_coco.py --data_dir . --splits train

# 转换到指定目录
python visdrone_to_coco.py --data_dir . --splits train val --output_dir ./coco_annotations
```

## 转换说明

### VisDrone格式
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

### COCO格式
生成的JSON文件包含标准的COCO格式结构：
- `info`: 数据集信息
- `licenses`: 许可证信息
- `images`: 图像信息列表
- `annotations`: 标注信息列表
- `categories`: 类别信息列表

### 类别映射

| VisDrone ID | 类别名称 | 是否包含 |
|------------|---------|----------|
| 0 | ignored_region | ❌ (跳过) |
| 1 | pedestrian | ✅ |
| 2 | people | ✅ |
| 3 | bicycle | ✅ |
| 4 | car | ✅ |
| 5 | van | ✅ |
| 6 | truck | ✅ |
| 7 | tricycle | ✅ |
| 8 | awning-tricycle | ✅ |
| 9 | bus | ✅ |
| 10 | motor | ✅ |
| 11 | others | ✅ |

### 过滤规则

转换过程中会自动过滤：
1. `ignored_region` (类别ID=0)
2. 被忽略的标注 (score=0)
3. 无效的边界框 (width≤0 或 height≤0)

## 依赖包

确保安装以下Python包：

```bash
pip install pillow tqdm
```

## 输出示例

转换完成后会显示统计信息：
```
Conversion complete!
Output saved to: visdrone_train_coco.json
Total images: 6471
Total annotations: 343204
Categories: 11
```

## 注意事项

1. 转换后的COCO文件保持原始的VisDrone类别ID，方便后续使用
2. 边界框格式为COCO标准的 [x, y, width, height]
3. 图像ID从1开始递增
4. 标注ID从1开始递增
5. 跳过了VisDrone中的忽略区域和无效标注 