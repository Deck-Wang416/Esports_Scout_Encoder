import torch
import numpy as np

from utils.config import load_cfg, load_yaml_ns, load_vocab_dir
from models.encoder import Encoder

def _build_env():
    # Load main config (including model/pooling/paths)
    cfg = load_cfg("configs/encoder.yaml")

    # Vocabulary sizes (used to construct Encoder)
    vocab = load_vocab_dir("configs/vocab")
    def _safe_len(ns_obj, attr):
        try:
            return len(getattr(getattr(ns_obj, attr), "tokens"))
        except Exception:
            return 0
    vocab_sizes = {
        "action":   _safe_len(vocab, "action"),
        "location": _safe_len(vocab, "location"),
        "outcome":  _safe_len(vocab, "outcome"),
        "impact":   _safe_len(vocab, "impact"),
        "weapon":   _safe_len(vocab, "weapon"),
    }

    # Merge tag vocab roles/traits into global vocab (only one PAD/UNK)
    tv = load_yaml_ns(getattr(getattr(cfg, "paths", object()), "tag_vocab", "configs/tags/tag_vocab.yaml"))
    roles  = getattr(getattr(tv, "roles", object()), "tokens", []) or []
    traits = getattr(getattr(tv, "traits", object()), "tokens", []) or []
    def _drop_pad_unk(toks): return [t for t in toks if t not in ("PAD", "UNK")]
    global_tag_tokens = ["PAD", "UNK"] + _drop_pad_unk(roles) + _drop_pad_unk(traits)
    token2id = {t: i for i, t in enumerate(global_tag_tokens)}
    tag_vocab_size = len(global_tag_tokens)
    return cfg, vocab_sizes, tag_vocab_size, token2id, roles, traits, global_tag_tokens

def _build_model(cfg, vocab_sizes, tag_vocab_size):
    torch.manual_seed(0)
    np.random.seed(0)
    model = Encoder(cfg, vocab_sizes=vocab_sizes, tag_vocab_size=tag_vocab_size)
    model.eval()
    return model

def test_encode_tag_determinism():
    cfg, vocab_sizes, tag_vocab_size, token2id, roles, traits, toks = _build_env()
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    # Prepare a valid set of IDs (including roles and traits)
    ids = []
    for t in toks[2:6]:  # Skip PAD/UNK, take 4 tokens
        ids.append(token2id[t])
    tag_ids = torch.tensor(ids, dtype=torch.long)

    with torch.no_grad():
        z1 = model.encode_tag(tag_ids)
        z2 = model.encode_tag(tag_ids)
    assert z1.shape == z2.shape and z1.dtype == z2.dtype
    assert torch.allclose(z1, z2, atol=0, rtol=0), "encode_tag is non-deterministic (inconsistent outputs for same input)"

def test_encode_tag_list_vs_tensor_equivalence():
    cfg, vocab_sizes, tag_vocab_size, token2id, roles, traits, toks = _build_env()
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    ids = [token2id[t] for t in toks[2:6]]  # Some valid IDs
    with torch.no_grad():
        z_list   = model.encode_tag(ids)  # List[int]
        z_tensor = model.encode_tag(torch.tensor(ids, dtype=torch.long))
    assert torch.allclose(z_list, z_tensor, atol=0, rtol=0), "List and Tensor inputs should produce identical encodings"

def test_encode_tag_mixed_ids_fallback_to_unk():
    cfg, vocab_sizes, tag_vocab_size, token2id, roles, traits, toks = _build_env()
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    UNK = token2id.get("UNK", 1)
    valid = token2id[toks[2]] if len(toks) > 2 else UNK
    mixed = torch.tensor([-3, valid, 10**9], dtype=torch.long)  # Negative + valid + out-of-range IDs

    with torch.no_grad():
        z_mixed = model.encode_tag(mixed)         # [3, D]
        z_unk   = model.encode_tag([UNK])         # [1, D]
        z_valid = model.encode_tag([valid])       # [1, D]

    # Positions 0 and 2 should match UNK encoding; position 1 should match valid encoding
    assert torch.allclose(z_mixed[0], z_unk[0]), "Negative ID not mapped to UNK"
    assert torch.allclose(z_mixed[2], z_unk[0]), "Out-of-range ID not mapped to UNK"
    assert torch.allclose(z_mixed[1], z_valid[0]), "Valid ID did not get correct encoding"

def test_encode_tag_empty_input_returns_empty():
    cfg, vocab_sizes, tag_vocab_size, *_ = _build_env()
    model = _build_model(cfg, vocab_sizes, tag_vocab_size)

    empty_list = []
    empty_tensor = torch.tensor([], dtype=torch.long)

    with torch.no_grad():
        z1 = model.encode_tag(empty_list)   # Expect shape [0, D]
        z2 = model.encode_tag(empty_tensor)

    D = getattr(cfg.model, "output_dim", None)
    assert z1.shape[0] == 0 and z2.shape[0] == 0, "Empty input should return tensor with batch dimension 0"
    if D is not None:
        assert z1.shape[1] == D and z2.shape[1] == D, "Output dimension D does not match config"
