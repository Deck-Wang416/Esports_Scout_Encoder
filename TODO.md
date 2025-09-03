# Encoder Part

---

## 1. Input Encoders
- [ ] Implement discrete embeddings: `action_emb`, `loc_emb`, `weapon_emb`, `team_emb`
- [ ] Implement multi-hot projections: `outcome_proj`, `impact_proj`
- [ ] Implement numeric projections: `timestamp_proj`, `damage_proj ([sum, mean, max, is_lethal])`
- [ ] Ensure all return `[B, T, H]` tensors

**Check:** Run `sanity_checks.py` → `encode_behavior -> (B, 128)` without errors

---

## 2. Feature Fusion
- [ ] Concatenate embeddings → Linear back to `H`
- [ ] Add `LayerNorm + Dropout`
- [ ] Output `[B, T, H]`

**Check:** `z_behavior mean≈0`, `var` non-zero & stable

---

## 3. Positional Encoding + Transformer
- [ ] Apply `SinCosPosEnc(H)`
- [ ] TransformerEncoder (`batch_first=True`, `src_key_padding_mask=~mask.bool()`)

**Check:** Still `[B, T, H]`, mask works correctly

---

## 4. Pooling & Behavior Head
- [ ] Implement pooling with multiple strategies (e.g., masked_mean, masked_max) → `[B, H]`
- [ ] Add `behavior_head (H→D)` to produce `z_behavior`

**Check:** `encode_behavior -> (B, D)` matches `output_dim` in config

---

## 5. Tag Encoder
- [ ] Add `tag_emb(|V_tags|, H)` + `tag_head(H→D)`
- [ ] Implement `encode_tag(tag_ids: List[int] or LongTensor[N]) → [N, D]`
- [ ] Add UNK fallback for out-of-range ids

**Check:** `encode_tag -> (N, D)` matches `z_behavior` dimension

---

## 6. Robustness
- [ ] Handle out-of-range / negative tag ids (map to UNK=1)
- [ ] Ensure correct dtypes (long/float/bool)
- [ ] Prevent divide-by-zero in pooling (`clamp_min(1.0)`)
- [ ] Construct effective mask (`m_eff`) to identify valid positions
- [ ] Handle all-padding cases by returning zero-safe outputs

**Check:** No crash on edge cases (empty tags, all padding, single sample)

---

## 7. Logging (Optional)
- [ ] Print key config params (H, D, layers, nhead) when building encoder
- [ ] Print tag vocab size/version

---

## 8. Documentation Alignment
- [ ] Verify 13 fields in **Batch Schema** are all consumed
- [ ] Ensure shapes in code match documentation
- [ ] Document optional fields (e.g. damage_mean, damage_max can be toggled off)

---

## Definition of Done
- `dump_vocab_stats.py` passes
- `sanity_checks.py` passes
- `encode_behavior(batch) -> [B,D]`
- `encode_tag(tag_ids) -> [N,D]`
- No NaN/Inf, not all-zero/constant
- Mask effective, deterministic outputs
- Docs and code consistent
