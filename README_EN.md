# ArisuIntelligence

<div align="center">

[![Version](https://img.shields.io/badge/version-v1.5.0-ff4f87)](https://github.com/XiaoliMEMZ/ArisuIntelligence/releases/tag/v1.5.0)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](./LICENSE)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.21-111f68)](https://docs.ultralytics.com/)

[简体中文](./README.md) | **English**

<img src="assets/kei.jpg" width="720" alt="Blue Archive Kei">

<sub>Illustration: <a href="https://www.pixiv.net/artworks/140524347">あすぱる先生 - "Kei-chan"</a> (used with permission)</sub>

</div>

## Overview

ArisuIntelligence provides object detection models trained for **RoboCupJunior Soccer Open** environments. The models detect the soccer ball, blue goal, yellow goal, and robot chassis on the field.

Version 1.5.0 is trained on the new **RCJ-Soccer-100K** dataset. It contains approximately 100,000 competition-scene images and was built with our in-house, fully automated annotation pipeline based on **Qwen3.5-27B**. Compared with [v1](https://github.com/XiaoliMEMZ/ArisuIntelligence/tree/v1), this release improves all four classes, behaves more consistently in difficult frames involving motion blur, occlusion, and lighting changes, and substantially reduces model weight size.

> [!NOTE]
> The RCJ-Soccer-100K dataset is not currently public. This repository releases inference weights only; it does not include training data, training code, or Hailo HEF files.

## Classes

| ID | Class | Description |
|---:|---|---|
| 0 | `Ball` | Soccer ball on the field |
| 1 | `BlueGoal` | Blue goal |
| 2 | `YellowGoal` | Yellow goal |
| 3 | `Chassis` | Robot chassis |

## Models and Performance

Both models are provided as production candidates for v1.5.0, with no single default recommendation. YOLO26s is smaller and achieves higher Precision and mAP50-95, while YOLOv8s achieves higher Recall and mAP50. Choose according to your deployment constraints and target metric.

| Model | Weights | Size | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26s | [`weights/yolo26s.pt`](./weights/yolo26s.pt) | 19.38 MiB | 26 | 0.92071 | 0.89375 | 0.95686 | 0.72946 |
| YOLOv8s | [`weights/yolov8s.pt`](./weights/yolov8s.pt) | 21.48 MiB | 50 | 0.90695 | 0.91841 | 0.95992 | 0.70501 |

Both runs used `imgsz=640` and were configured for up to 50 epochs. "Best epoch" is the epoch that produced the best checkpoint. These metrics compare the two models from this release under the current validation setup and have not been independently reproduced. Because no v1 results from the same validation set are available, this project does not present a direct numerical comparison between v1 and v1.5.0.

## Quick Start

### Install

Python 3.10 or later is recommended:

```bash
python -m pip install "ultralytics>=8.4.21"
```

### Python

```python
from ultralytics import YOLO

model = YOLO("weights/yolo26s.pt")
results = model("input.jpg", conf=0.5, imgsz=640)

for result in results:
    print(result.boxes)
    result.save(filename="output.jpg")
```

Replace the model path with `weights/yolov8s.pt` to use YOLOv8s.

### CLI

```bash
yolo predict \
  model=weights/yolo26s.pt \
  source=input.jpg \
  imgsz=640 \
  conf=0.5
```

Predictions are saved to `runs/detect/predict/` by default.

## Verify Downloads

Checksums are recorded in [`weights/SHA256SUMS`](./weights/SHA256SUMS):

```bash
cd weights
shasum -a 256 -c SHA256SUMS
```

The weights and checksum file are also available separately from the [v1.5.0 Release](https://github.com/XiaoliMEMZ/ArisuIntelligence/releases/tag/v1.5.0).

## Version History

- **v1.5.0**: YOLO26s and YOLOv8s models trained on RCJ-Soccer-100K.
- **[v1](https://github.com/XiaoliMEMZ/ArisuIntelligence/tree/v1)**: Original YOLOv8s/Hailo release, preserved on a dedicated branch.

## License and Credits

Original content in this repository is released under the [MIT License](./LICENSE). The model weights include third-party Ultralytics components and license notices; see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for details.
