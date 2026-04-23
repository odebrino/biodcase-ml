import argparse
import shutil
from pathlib import Path


def cache_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return list(root.rglob("*.pt"))


def cache_size(files: list[Path]) -> int:
    return sum(path.stat().st_size for path in files if path.exists())


def print_summary(root: Path) -> None:
    files = cache_files(root)
    size_gb = cache_size(files) / 1024**3
    print(f"root: {root}")
    print(f"files: {len(files)}")
    print(f"size_gb: {size_gb:.2f}")


def clear_cache(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    print(f"cleared: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or clear the spectrogram tensor cache.")
    parser.add_argument("--root", default="processed_cache")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--clear-except-current", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if args.clear_except_current:
        raise SystemExit("--clear-except-current is not implemented safely yet. Use --clear to remove the full cache.")
    if args.clear:
        clear_cache(root)
    else:
        print_summary(root)


if __name__ == "__main__":
    main()
