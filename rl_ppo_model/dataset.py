#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B2D数据集加载器 - 用于RL训练
读取图像和专家动作
"""

import os
import glob
import gzip
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import tarfile
from pathlib import Path


class B2DDataset(Dataset):
    """
    B2D数据集加载器

    数据结构：
    - camera/rgb_front/*.jpg - RGB前置相机图像
    - anno/*.json.gz - 专家动作和车辆状态
    """

    def __init__(self, data_root, image_size=(224, 224), max_clips=None):
        """
        Args:
            data_root: 数据根目录（包含解压后的clips）
            image_size: 图像resize大小
            max_clips: 最大使用的clips数量（None=使用全部）
        """
        self.data_root = Path(data_root)
        self.image_size = image_size

        print(f"🔍 正在扫描数据集: {data_root}")

        # 查找所有已解压的clip目录
        self.clip_dirs = []
        for clip_dir in sorted(self.data_root.glob("*")):
            if clip_dir.is_dir() and not clip_dir.name.endswith('.tar.gz'):
                # 检查是否有必要的子目录
                if (clip_dir / "camera" / "rgb_front").exists() and \
                   (clip_dir / "anno").exists():
                    self.clip_dirs.append(clip_dir)

        if max_clips:
            self.clip_dirs = self.clip_dirs[:max_clips]

        print(f"✅ 找到 {len(self.clip_dirs)} 个clips")

        # 构建所有帧的索引
        self.samples = []
        for clip_dir in self.clip_dirs:
            # 获取该clip的所有帧
            anno_files = sorted((clip_dir / "anno").glob("*.json.gz"))

            for anno_file in anno_files:
                # anno_file.stem 返回 "00000.json"，需要去掉 .json
                frame_id = anno_file.stem.replace('.json', '')  # 例如 "00000"
                img_file = clip_dir / "camera" / "rgb_front" / f"{frame_id}.jpg"

                if img_file.exists():
                    self.samples.append({
                        'clip_dir': clip_dir,
                        'frame_id': frame_id,
                        'img_file': img_file,
                        'anno_file': anno_file
                    })

        print(f"✅ 总共 {len(self.samples)} 个训练样本")

        if len(self.samples) == 0:
            raise ValueError("❌ 没有找到任何训练样本！请检查数据是否已解压。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. 读取图像
        img = Image.open(sample['img_file']).convert('RGB')
        img = img.resize(self.image_size)
        img = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        # 2. 读取标注（专家动作和状态）
        with gzip.open(sample['anno_file'], 'rt') as f:
            anno = json.load(f)

        # 3. 提取状态（作为RL的observation）
        state = torch.tensor([
            anno['speed'],           # 速度
            anno['x'],               # X坐标
            anno['y'],               # Y坐标
            anno['theta'],           # 航向角
            anno['x_command_far'],   # 目标X
            anno['y_command_far'],   # 目标Y
        ], dtype=torch.float32)

        # 4. 提取专家动作（作为RL的expert action）
        action = torch.tensor([
            anno['throttle'],  # 油门 [0, 1]
            anno['steer'],     # 转向 [-1, 1]
            anno['brake'],     # 刹车 [0, 1]
        ], dtype=torch.float32)

        return {
            'image': img,           # (3, H, W)
            'state': state,         # (6,)
            'action': action,       # (3,) - 专家动作
            'frame_id': sample['frame_id'],
            'clip_name': sample['clip_dir'].name
        }


def create_dataloader(data_root, batch_size=32, image_size=(224, 224),
                     num_workers=4, shuffle=True, max_clips=None):
    """
    创建数据加载器

    Args:
        data_root: 数据根目录
        batch_size: batch大小
        image_size: 图像大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        max_clips: 最大clips数量

    Returns:
        DataLoader
    """
    dataset = B2DDataset(
        data_root=data_root,
        image_size=image_size,
        max_clips=max_clips
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    print("="*70)
    print("🧪 测试B2D数据加载器")
    print("="*70)
    print()

    # 数据路径（根据你的实际情况修改）
    data_root = "/home/ajifang/b2drive/Bench2Drive-RL50GB/datasets--rethinklab--Bench2Drive/snapshots"

    # 找到snapshot目录
    import glob
    snapshot_dirs = glob.glob(f"{data_root}/*")
    if snapshot_dirs:
        data_root = snapshot_dirs[0]
        print(f"📂 数据目录: {data_root}")
    else:
        print("❌ 找不到snapshot目录")
        exit(1)

    try:
        # 创建数据加载器（先测试10个clips）
        dataloader = create_dataloader(
            data_root=data_root,
            batch_size=4,
            image_size=(224, 224),
            num_workers=2,
            shuffle=True,
            max_clips=10  # 测试时只用10个clips
        )

        print()
        print("="*70)
        print("📊 数据集信息:")
        print(f"   - 总样本数: {len(dataloader.dataset)}")
        print(f"   - Batch数量: {len(dataloader)}")
        print(f"   - Batch大小: {dataloader.batch_size}")
        print("="*70)
        print()

        # 测试加载一个batch
        print("🔄 测试加载一个batch...")
        batch = next(iter(dataloader))

        print(f"✅ 加载成功!")
        print(f"   - image shape: {batch['image'].shape}")
        print(f"   - state shape: {batch['state'].shape}")
        print(f"   - action shape: {batch['action'].shape}")
        print()

        print("📋 示例数据:")
        print(f"   - Speed: {batch['state'][0, 0]:.4f} m/s")
        print(f"   - Throttle: {batch['action'][0, 0]:.4f}")
        print(f"   - Steer: {batch['action'][0, 1]:.4f}")
        print(f"   - Brake: {batch['action'][0, 2]:.4f}")
        print()

        print("="*70)
        print("🎉 数据加载器测试成功!")
        print("="*70)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
