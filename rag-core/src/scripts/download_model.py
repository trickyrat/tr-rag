"""Download HuggingFace models via ModelScope mirror (faster in China).

Usage:
    uv run download-model BAAI/bge-m3
    uv run download-model BAAI/bge-m3 -d D:/models
    uv run download-model BAAI/bge-reranker-v2-m3
    uv run download-model --list   # show known models
"""
import argparse
import sys
from pathlib import Path

from modelscope import snapshot_download

# Known models used by this project
_KNOWN_MODELS: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
    "harrier-6b": "microsoft/harrier-oss-v1-0___6b",
}


def main():
    parser = argparse.ArgumentParser(description="Download models via ModelScope")
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (e.g. BAAI/bge-m3) or shortcut (e.g. bge-m3)",
    )
    parser.add_argument(
        "-d", "--cache-dir",
        default="D:/models",
        help="Local directory to save models (default: D:/models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known model shortcuts",
    )
    args = parser.parse_args()

    if args.list:
        print("Known model shortcuts:")
        for alias, full in _KNOWN_MODELS.items():
            print(f"  {alias:25s} → {full}")
        return 0

    if not args.model:
        parser.print_help()
        return 1

    # Resolve shortcut
    model_id = _KNOWN_MODELS.get(args.model, args.model)

    target = str(Path(args.cache_dir).resolve())
    print(f"Downloading {model_id} → {target} ...")
    try:
        snapshot_download(model_id, cache_dir=target)
        print(f"Done: {model_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    main()