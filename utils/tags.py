from utils.config import load_yaml


def load_tag_vocab(path: str):
    """
    Load tag vocab YAML and return basic stats.
    """
    y = load_yaml(path)
    roles  = y.get("roles", [])
    traits = y.get("traits", [])

    def drop_pad_unk(tokens):
        return [t for t in tokens if t not in ("PAD", "UNK")]

    size = len(drop_pad_unk(roles)) + len(drop_pad_unk(traits))

    return {
        "version": y.get("version", "unknown"),
        "size_hint": size
    }
