# PJ2 项目 - CIFAR-10 & BatchNorm 实验

## 项目概述

复旦深度学习课程 PJ2，截止 2026-06-14。两个任务：
- 任务 1 (60%)：CIFAR-10 训练网络，对比不同结构/优化器/激活函数/正则化
- 任务 2 (30%)：VGG-A vs VGG-A+BN 性能对比 + loss landscape 分析

GitHub: https://github.com/CZcoco/Fdu_DL_2.git

## 当前状态

### 已完成
- 项目代码框架搭建完毕，主脚本 `VGG_Loss_Landscape.py` 支持：
  - `--mode train`：模型对比训练
  - `--mode optim`：优化器对比（Adam/AdamW/SGD on vgg_a_bn）
  - `--mode landscape`：loss landscape 实验（vgg_a vs vgg_a_bn 多学习率）
  - `--mode all`：以上全部
- 可配置优化器（adam/adamw/sgd）、cosine schedule、label smoothing、weight decay、数据增强
- 4 个模型变体：VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light
- CIFAR-10 数据集已验证可自动下载

### 已发现的问题
- **vgg_a 和 vgg_a_dropout 在 lr=1e-3 下完全不收敛**（val acc = 10%，随机猜）
- vgg_a_bn 在 lr=1e-3 下正常训练到 85.2%
- vgg_a_light（浅网络）在 lr=1e-3 下正常训练到 73.5%
- 原因：深层网络无 BN 时梯度不稳定，高学习率下无法收敛
- **这恰好是 BN 论文的核心论点**，可以作为报告素材

### 待完成实验

1. **让 vgg_a / vgg_a_dropout 也能正常训练**：
   - 用更低学习率 (1e-4 或 5e-5)
   - 或用 SGD + momentum (lr=0.01, momentum=0.9)
   - 目标：让所有模型都能收敛，做公平对比

2. **两组对比实验设计**：
   - 高 lr 组 (1e-3)：展示 BN 允许使用更高学习率（BN 能训，non-BN 不能）
   - 低 lr 组 (1e-4)：所有模型都能训练，做公平的结构/正则化对比

3. **优化器对比** (`--mode optim`)：
   - 在 vgg_a_bn 上比较 Adam / AdamW / SGD
   - 还没跑过

4. **Loss landscape 实验** (`--mode landscape`)：
   - vgg_a vs vgg_a_bn 在多个学习率下训练
   - 画 min/max loss band 图（类似论文 Figure）
   - 学习率范围需要调整：BN 模型可以用 [1e-4, 1e-3, 2e-3, 5e-3]，non-BN 模型只能用 [1e-5, 5e-5, 1e-4, 5e-4]

5. **额外需要的实验**（项目要求）：
   - 不同激活函数对比（ReLU / LeakyReLU / ELU）—— 代码还没支持，需要加
   - 梯度范数分析（代码已记录 gradient_norms，需要画图）
   - 训练速度对比（每 epoch 时间）

## 运行方式

```bash
cd codes/VGG_BatchNorm
pip install torch torchvision tqdm matplotlib numpy

# 全部实验一次跑
python VGG_Loss_Landscape.py --mode all --epochs 20 --landscape-epochs 20 \
  --n-items -1 --val-items -1 --augment --lr 0.001 --batch-size 128 \
  --landscape-lrs 1e-3 2e-3 5e-4 1e-4 5e-3 --num-workers 4

# 单独跑某个模式
python VGG_Loss_Landscape.py --mode train --models vgg_a_bn --epochs 30 \
  --n-items -1 --val-items -1 --augment --optimizer sgd --lr 0.01
```

## 关键参数说明

- `--n-items -1`：使用完整训练集 (50000)
- `--val-items -1`：使用完整测试集 (10000)
- `--augment`：开启 RandomCrop + RandomHorizontalFlip
- `--num-workers 4`：服务器上建议开，本地 Windows 用 0

## 输出结构

```
reports/
├── figures/          # 训练曲线图、loss landscape 图、样本图
├── models/           # .pt 模型权重
└── results/          # .csv 训练历史、.json 实验摘要
```

## 代码结构

```
codes/VGG_BatchNorm/
├── VGG_Loss_Landscape.py   # 主训练/实验脚本
├── models/vgg.py            # VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light
├── data/loaders.py          # CIFAR-10 数据加载
└── utils/nn.py              # 权重初始化
```

## 建议的下一步改动

1. `run_single_training` 中对不同模型自动选择合适学习率，或新增参数支持 per-model lr
2. 添加激活函数可配置（在 vgg.py 中把 ReLU 参数化）
3. 添加梯度范数可视化函数
4. 增加训练时间记录（每 epoch wall-clock time）
5. landscape 实验中对 non-BN 模型用更低的学习率范围

## 报告要求摘要

- 中文报告
- 必须包含：姓名、学号、GitHub 链接、数据集链接、模型权重链接
- 任务 1：网络结构、参数量、最佳 test error、训练速度、可视化
- 任务 2：VGG-A vs VGG-A+BN 性能对比 + loss landscape + 梯度稳定性分析
