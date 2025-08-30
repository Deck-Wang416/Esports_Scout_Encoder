# Esports_Scout_Encoder
Minimal runnable encoder project for esports data.  
This repo currently provides two test scripts to validate the setup.

## 1. Check vocabulary files

Run the vocab dump script to verify vocabularies (`configs/vocab/*.yaml` and `configs/tags/tag_vocab.yaml`) are valid:

```bash
python scripts/dump_vocab_stats.py \
  --vocab-dir configs/vocab \
  --tag-vocab configs/tags/tag_vocab.yaml
```

Expected: it will print the size of each vocab, warn if PAD/UNK are missing, and report errors if tokens are duplicated.

## 2. Run end-to-end sanity checks

Run the minimal pipeline test with encoder + mock data:

```bash
export PYTHONPATH=$(pwd)
python scripts/sanity_checks.py \
  --encoder-cfg configs/encoder.yaml \
  --dataloader-cfg configs/dataloader.yaml \
  --json data/mock/sample_v01.json
```

Expected: it will
	• Load mock JSON into a batch,
	• Run the encoder forward pass (encode_behavior and encode_tag),
	• Print dimension / numeric / consistency checks,
	• End with Sanity checks finished without assertion errors.
