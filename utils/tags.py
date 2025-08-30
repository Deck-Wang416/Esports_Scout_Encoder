from utils.config import load_yaml
def load_tag_vocab(path):
    y = load_yaml(path)
    roles  = y.get("roles", [])
    traits = y.get("traits", [])
    size = len(roles) + len(traits) - 2 - 2
    return {"version": y.get("version", "unknown"), "size_hint": size}
