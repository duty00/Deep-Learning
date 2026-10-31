"""Run after editable installation: python scripts/evaluate.py --config configs/baseline.json"""
import sys
from dl_project.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
