import argparse
import os
import sys
import glob
import yaml
from typing import List, Tuple

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _extract_tokens(node) -> List[str]:
    """
    兼容两种格式：
    1) {"tokens": ["PAD","UNK","..."]}
    2) ["PAD","UNK","..."]
    """
    if isinstance(node, dict) and "tokens" in node:
        return list(node["tokens"])
    if isinstance(node, list):
        return list(node)
    raise ValueError("YAML 格式不支持：需要 'tokens' 列表或顶层即为列表。")

def check_vocab(name: str, tokens: List[str]) -> Tuple[List[str], List[str]]:
    errors, warns = [], []
    if not tokens:
        errors.append(f"{name}: 词表为空。")
        return errors, warns

    # 基本检查
    if "PAD" not in tokens:
        errors.append(f"{name}: 缺少 'PAD'。")
    if "UNK" not in tokens:
        errors.append(f"{name}: 缺少 'UNK'。")

    # 索引检查（不强制但推荐）
    if "PAD" in tokens and tokens.index("PAD") != 0:
        warns.append(f"{name}: 建议 'PAD' 位于 index 0（当前 index={tokens.index('PAD')}).")
    if "UNK" in tokens and tokens.index("UNK") != 1:
        warns.append(f"{name}: 建议 'UNK' 位于 index 1（当前 index={tokens.index('UNK')}).")

    # 重复检查
    seen = set()
    dups = set()
    for t in tokens:
        if t in seen:
            dups.add(t)
        seen.add(t)
    if dups:
        errors.append(f"{name}: 存在重复 token: {sorted(dups)}")

    return errors, warns

def print_header(title: str):
    print(f"\n{GREEN}== {title} =={RESET}")

def main():
    ap = argparse.ArgumentParser(description="Dump basic stats for vocab and tag vocab.")
    ap.add_argument("--vocab-dir", required=True, help="目录，例如 configs/vocab")
    ap.add_argument("--tag-vocab", required=False, help="文件，例如 configs/tags/tag_vocab.yaml")
    ap.add_argument("--strict", action="store_true", help="出现错误时以非零码退出")
    args = ap.parse_args()

    any_error = False

    # ---- 普通词表目录 ----
    print_header("Vocab Stats (action/location/outcome/impact/weapon)")
    vocab_files = sorted(glob.glob(os.path.join(args.vocab_dir, "*.yaml")))
    if not vocab_files:
        print(f"{YELLOW}警告：未在 {args.vocab_dir} 找到任何 *.yaml 词表文件。{RESET}")

    for vf in vocab_files:
        try:
            data = load_yaml(vf)
            tokens = _extract_tokens(data)
            errors, warns = check_vocab(os.path.basename(vf), tokens)

            print(f"- {os.path.basename(vf)}: size={len(tokens)}")
            if warns:
                for w in warns:
                    print(f"  {YELLOW}warn:{RESET} {w}")
            if errors:
                any_error = True
                for e in errors:
                    print(f"  {RED}error:{RESET} {e}")
        except Exception as e:
            any_error = True
            print(f"{RED}error:{RESET} 解析失败 {vf}: {e}")

    # ---- Tag 词表文件 ----
    if args.tag_vocab:
        print_header("Tag Vocab Stats (roles / traits)")
        try:
            tv = load_yaml(args.tag_vocab)

            # 支持两种层级结构：
            # 1) 顶层含 roles: {tokens:[...]} / traits: {tokens:[...]}
            # 2) 顶层直接 tokens:[...]（视为全局词表）
            roles_tokens, traits_tokens = [], []
            if isinstance(tv, dict) and ("roles" in tv or "traits" in tv):
                if "roles" in tv:
                    roles_tokens = _extract_tokens(tv["roles"])
                if "traits" in tv:
                    traits_tokens = _extract_tokens(tv["traits"])
            else:
                # 兼容：整个 tag_vocab 就是一个 tokens 列表
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

            # 版本字段（可选）
            version = None
            for k in ("version", "schema_version"):
                if isinstance(tv, dict) and k in tv:
                    version = tv[k]
            if version:
                print(f"- tag vocab version: {version}")

        except Exception as e:
            any_error = True
            print(f"{RED}error:{RESET} 解析失败 {args.tag_vocab}: {e}")

    # ---- 总结与退出码 ----
    print_header("Summary")
    if any_error:
        msg = "发现错误（见上方 error）。"
        if args.strict:
            print(f"{RED}{msg} 由于 --strict，退出码 1。{RESET}")
            sys.exit(1)
        else:
            print(f"{YELLOW}{msg} 未使用 --strict，不改变退出码。{RESET}")
    else:
        print(f"{GREEN}未发现错误。{RESET}")

if __name__ == "__main__":
    main()
