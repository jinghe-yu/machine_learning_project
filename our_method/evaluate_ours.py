"""Evaluate memory-based rejection sampling on official Boolean dataset JSON."""

from benchmark_cli import run_cli


def main() -> None:
    run_cli(__doc__, baseline=False)


if __name__ == "__main__":
    main()
