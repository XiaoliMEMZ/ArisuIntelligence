# ArisuIntelligence

<div align="center">

[![Version](https://img.shields.io/badge/version-v1.5.0-ff4f87)](https://github.com/XiaoliMEMZ/ArisuIntelligence/releases/tag/v1.5.0)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](./LICENSE)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.21-111f68)](https://docs.ultralytics.com/)

**简体中文** | [English](./README_EN.md)

> 又是为了爱丽丝吗...

<img src="assets/kei.jpg" width="720" alt="Blue Archive Kei">

<sub>插画：<a href="https://www.pixiv.net/artworks/140524347">あすぱる先生 - 《ケイちゃん》</a></sub>

</div>

## 项目简介

ArisuIntelligence 是面向 **RoboCupJunior Soccer Open** 比赛场景训练的目标检测模型，能够识别场内足球、蓝色球门、黄色球门和机器人底盘。

v1.5.0 基于全新的 **RCJ-Soccer-100K** 数据集训练。该数据集包含约 10 万张比赛场景图片，由我们自研的、基于 **Qwen3.5-27B** 的全自动化标注方案构建。与 [v1](https://github.com/XiaoliMEMZ/ArisuIntelligence/tree/v1) 相比，新版本在四个类别上均有整体提升，并在运动模糊、目标遮挡和光照变化等困难画面中表现得更加稳定，同时显著缩小了模型权重体积。

> [!NOTE]
> RCJ-Soccer-100K 数据集当前不公开。本仓库仅发布推理权重，不包含训练数据、训练代码或 Hailo HEF 文件。

## 可识别类别

| ID | 类别 | 说明 |
|---:|---|---|
| 0 | `Ball` | 场内足球 |
| 1 | `BlueGoal` | 蓝色球门 |
| 2 | `YellowGoal` | 黄色球门 |
| 3 | `Chassis` | 机器人底盘 |

## 模型与性能

两个模型均作为 v1.5.0 的生产候选提供，不设置单一推荐模型。YOLO26s 文件更小，并取得更高的 Precision 与 mAP50-95；YOLOv8s 则取得更高的 Recall 与 mAP50，可根据部署环境和侧重指标选择。

| 模型 | 权重 | 大小 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26s | [`weights/yolo26s.pt`](./weights/yolo26s.pt) | 19.38 MiB | 26 | 0.92071 | 0.89375 | 0.95686 | 0.72946 |
| YOLOv8s | [`weights/yolov8s.pt`](./weights/yolov8s.pt) | 21.48 MiB | 50 | 0.90695 | 0.91841 | 0.95992 | 0.70501 |

两次训练均使用 `imgsz=640`，最多训练 50 epochs。表中的“最佳 epoch”表示最佳 checkpoint 出现的轮次。以上指标用于比较本版本两个模型在当前验证配置下的表现，并非独立第三方复现结果；由于缺少同一验证集上的旧版指标，本项目不对 v1 与 v1.5.0 作数值上的直接比较。

## 快速上手

### 安装依赖

建议使用 Python 3.10 或更高版本：

```bash
python -m pip install "ultralytics>=8.4.21"
```

### Python 推理

```python
from ultralytics import YOLO

model = YOLO("weights/yolo26s.pt")
results = model("input.jpg", conf=0.5, imgsz=640)

for result in results:
    print(result.boxes)
    result.save(filename="output.jpg")
```

将模型路径替换为 `weights/yolov8s.pt` 即可使用 YOLOv8s。

### CLI 推理

```bash
yolo predict \
  model=weights/yolo26s.pt \
  source=input.jpg \
  imgsz=640 \
  conf=0.5
```

推理结果默认保存在 `runs/detect/predict/`。

## 文件校验

校验值记录在 [`weights/SHA256SUMS`](./weights/SHA256SUMS)：

```bash
cd weights
shasum -a 256 -c SHA256SUMS
```

也可以从 [v1.5.0 Release](https://github.com/XiaoliMEMZ/ArisuIntelligence/releases/tag/v1.5.0) 单独下载权重和校验文件。

## 版本历史

- **v1.5.0**：基于 RCJ-Soccer-100K 发布 YOLO26s 与 YOLOv8s 双模型。
- **[v1](https://github.com/XiaoliMEMZ/ArisuIntelligence/tree/v1)**：原始 YOLOv8s/Hailo 版本，保留在独立分支中。

## 许可证与致谢

仓库中的自有内容以 [MIT License](./LICENSE) 发布。模型权重包含 Ultralytics 的第三方组件及其许可声明，详情见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
