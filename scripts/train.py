"""Run after editable installation: python scripts/train.py --config configs/baseline.json"""
import sys
from dl_project.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
