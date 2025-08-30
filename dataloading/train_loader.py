import yaml
from torch.utils.data import DataLoader
from .collate import collate_fn

def build_loaders(train_ds, val_ds, cfg_path="configs/dataloader.yaml"):
    """Build train/val DataLoaders from datasets and config YAML."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "loader" in cfg, "Missing 'loader' section in dataloader.yaml"

    bs  = cfg["loader"]["batch_size"]
    nw  = cfg["loader"]["num_workers"]
    tr_shuffle  = cfg["loader"]["shuffle_train"]
    val_shuffle = cfg["loader"]["shuffle_val"]

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=tr_shuffle, num_workers=nw, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,   batch_size=bs, shuffle=val_shuffle, num_workers=nw, collate_fn=collate_fn
    )
    return train_loader, val_loader
