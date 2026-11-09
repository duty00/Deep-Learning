from .cli import main

# Guard needed on Windows: DataLoader workers (spawn) re-import this module.
if __name__ == "__main__":
    raise SystemExit(main())
