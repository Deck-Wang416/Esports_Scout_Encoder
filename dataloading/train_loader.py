import yaml
from torch.utils.data import DataLoader
from .collate import collate_fn

def build_loaders(train_ds, val_ds, cfg_path="configs/dataloader.yaml"):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    bs  = cfg["loader"]["batch_size"]
    nw  = cfg["loader"]["num_workers"]
    tr_shuffle = cfg["loader"]["shuffle_train"]
    val_shuffle= cfg["loader"]["shuffle_val"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=tr_shuffle, num_workers=nw, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=val_shuffle, num_workers=nw, collate_fn=collate_fn)
    return train_loader, val_loader
