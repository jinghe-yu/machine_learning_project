"""Run a fixed-budget Boolean-expression baseline without sampling history."""

from benchmark_cli import run_cli


def main() -> None:
    run_cli(__doc__, baseline=True)


if __name__ == "__main__":
    main()
