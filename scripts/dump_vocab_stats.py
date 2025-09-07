import argparse
import glob
import os
import sys
from typing import List, Tuple

import yaml

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_tokens(node) -> List[str]:
    """
    Accept two forms:
      1) {"tokens": ["PAD", "UNK", "..."]}
      2) ["PAD", "UNK", "..."]
    """
    if isinstance(node, dict) and "tokens" in node:
        return list(node["tokens"])
    if isinstance(node, list):
        return list(node)
    raise ValueError("Unsupported YAML structure: need a 'tokens' list or a top-level list.")


def check_vocab(name: str, tokens: List[str]) -> Tuple[List[str], List[str]]:
    errors, warns = [], []
    if not tokens:
        errors.append(f"{name}: empty vocabulary.")
        return errors, warns

    # Required special tokens
    if "PAD" not in tokens:
        errors.append(f"{name}: missing 'PAD'.")
    if "UNK" not in tokens:
        errors.append(f"{name}: missing 'UNK'.")

    # Recommended indices (not enforced)
    if "PAD" in tokens and tokens.index("PAD") != 0:
        warns.append(f"{name}: recommended 'PAD' at index 0 (current {tokens.index('PAD')}).")
    if "UNK" in tokens and tokens.index("UNK") != 1:
        warns.append(f"{name}: recommended 'UNK' at index 1 (current {tokens.index('UNK')}).")

    # Duplicate detection
    seen, dups = set(), set()
    for t in tokens:
        if t in seen:
            dups.add(t)
        seen.add(t)
    if dups:
        errors.append(f"{name}: duplicated tokens: {sorted(dups)}")

    return errors, warns


def print_header(title: str):
    print(f"\n{GREEN}== {title} =={RESET}")


def main():
    ap = argparse.ArgumentParser(description="Dump basic stats for vocab and tag vocab.")
    ap.add_argument("--vocab-dir", required=True, help="Directory, e.g. configs/vocab")
    ap.add_argument("--tag-vocab", required=False, help="File, e.g. configs/tags/tag_vocab.yaml")
    ap.add_argument("--strict", action="store_true", help="Exit with non-zero status on errors")
    args = ap.parse_args()

    any_error = False

    # ---- Standard vocabs (action/location/outcome/impact/weapon) ----
    print_header("Vocab Stats (action/location/outcome/impact/weapon)")
    vocab_files = sorted(glob.glob(os.path.join(args.vocab_dir, "*.yaml")))
    if not vocab_files:
        print(f"{YELLOW}warn:{RESET} no *.yaml files found under {args.vocab_dir}")

    for vf in vocab_files:
        try:
            data = load_yaml(vf)
            tokens = _extract_tokens(data)
            errors, warns = check_vocab(os.path.basename(vf), tokens)

            print(f"- {os.path.basename(vf)}: size={len(tokens)}")
            for w in warns:
                print(f"  {YELLOW}warn:{RESET} {w}")
            for e in errors:
                any_error = True
                print(f"  {RED}error:{RESET} {e}")
        except Exception as e:
            any_error = True
            print(f"{RED}error:{RESET} failed to parse {vf}: {e}")

    # ---- Tag vocab (roles / traits) ----
    if args.tag_vocab:
        print_header("Tag Vocab Stats (roles / traits)")
        try:
            tv = load_yaml(args.tag_vocab)

            # Accept two structures:
            #   1) top-level roles:{tokens:[...]} / traits:{tokens:[...]}
            #   2) top-level list (treated as a global tag list)
            roles_tokens, traits_tokens = [], []
            if isinstance(tv, dict) and ("roles" in tv or "traits" in tv):
                if "roles" in tv:
                    roles_tokens = _extract_tokens(tv["roles"])
                if "traits" in tv:
                    traits_tokens = _extract_tokens(tv["traits"])
            else:
                roles_tokens = _extract_tokens(tv)

            if roles_tokens:
                errs, warns = check_vocab("roles", roles_tokens)
                print(f"- roles: size={len(roles_tokens)}")
                for w in warns:
                    print(f"  {YELLOW}warn:{RESET} {w}")
                for e in errs:
                    any_error = True
                    print(f"  {RED}error:{RESET} {e}")

            if traits_tokens:
                errs, warns = check_vocab("traits", traits_tokens)
                print(f"- traits: size={len(traits_tokens)}")
                for w in warns:
                    print(f"  {YELLOW}warn:{RESET} {w}")
                for e in errs:
                    any_error = True
                    print(f"  {RED}error:{RESET} {e}")

            # Optional version field
            version = None
            if isinstance(tv, dict):
                version = tv.get("version") or tv.get("schema_version")
            if version:
                print(f"- tag vocab version: {version}")

        except Exception as e:
            any_error = True
            print(f"{RED}error:{RESET} failed to parse {args.tag_vocab}: {e}")

    print_header("Summary")
    if any_error:
        msg = "Errors found (see above)."
        if args.strict:
            print(f"{RED}{msg} Exiting with code 1 due to --strict.{RESET}")
            sys.exit(1)
        else:
            print(f"{YELLOW}{msg} Not strict; exit code remains 0.{RESET}")
    else:
        print(f"{GREEN}No errors detected.{RESET}")


if __name__ == "__main__":
    main()
