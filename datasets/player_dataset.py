import random
from torch.utils.data import Dataset

class PlayerDataset(Dataset):
    """Dataset wrapper: parse JSON files via adapter, split into train/val/test."""

    def __init__(self, json_files, adapter, split="train", ratio=0.8, seed=42, shuffle_train=True):
        self.samples = []
        for f in json_files:
            self.samples.extend(adapter.parse_file(f))

        if split == "train" and shuffle_train:
            random.seed(seed)
            random.shuffle(self.samples)

        # split by ratio
        n = int(len(self.samples) * ratio)
        if split == "train":
            self.samples = self.samples[:n]
        elif split in ("val", "test"):
            self.samples = self.samples[n:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
