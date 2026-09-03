"""Remove rebuildable factor artifacts outside an approved candidate list."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-root", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    keep = ids(args.candidate_file)
    targets = [path.parent.parent for path in args.factor_root.glob("*_qfq/v1/factor.parquet") if f"{path.parent.parent.name}_v1" not in keep]
    print(f"keep={len(keep)} remove={len(targets)}")
    if args.apply:
        for path in targets:
            shutil.rmtree(path)


if __name__ == "__main__":
    main()
