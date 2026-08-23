"""Inspect the modalities that were downloaded for a single Ego-Exo4D take."""

import argparse
import json

from egomemory.preprocessing.take_loader import inspect_take


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("take_dir", help="Path to one downloaded take")
    args = parser.parse_args()
    print(json.dumps(inspect_take(args.take_dir), indent=2))


if __name__ == "__main__":
    main()
