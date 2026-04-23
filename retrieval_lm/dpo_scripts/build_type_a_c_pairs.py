#!/usr/bin/env python

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="train_rl_dpo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).with_name("build_dpo_pairs.py")
    jobs = [
        ("type_a", output_dir / f"{args.prefix}_type_a.jsonl"),
        ("type_c", output_dir / f"{args.prefix}_type_c.jsonl"),
    ]

    for pair_type, output_file in jobs:
        cmd = [
            sys.executable,
            str(script_path),
            "--input-file",
            args.input_file,
            "--output-file",
            str(output_file),
            "--pair-type",
            pair_type,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
