# datasets/player_dataset.py
import random
from torch.utils.data import Dataset

class PlayerDataset(Dataset):
    def __init__(self, json_files, adapter, split="train", ratio=0.8, seed=42, shuffle_train=True):
        self.samples = []
        for f in json_files:
            self.samples.extend(adapter.parse_file(f))

        # 先随机打乱（只对 train）
        if split == "train" and shuffle_train:
            random.seed(seed)
            random.shuffle(self.samples)

        # 按比例划分
        n = int(len(self.samples) * ratio)
        if split == "train":
            self.samples = self.samples[:n]
        else:
            self.samples = self.samples[n:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
