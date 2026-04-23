import argparse
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model. Optionally load weights from an accelerate epoch checkpoint."
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help="Path to the base model.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the PEFT adapter folder saved with save_pretrained (top-level output dir).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to write the merged model.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional accelerate checkpoint dir such as epoch_0. If provided, weights are loaded from this checkpoint before merging.",
    )
    return parser.parse_args()


def find_weight_file(checkpoint_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model weight file found in {checkpoint_dir}")


def load_state_dict(weight_file: Path):
    if weight_file.suffix == ".safetensors":
        return load_file(str(weight_file))
    return torch.load(weight_file, map_location="cpu")


def main():
    args = parse_args()

    adapter_path = Path(args.adapter_path)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype="auto")
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    if checkpoint_dir is not None:
        weight_file = find_weight_file(checkpoint_dir)
        state_dict = load_state_dict(weight_file)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint weights from: {weight_file}")
        print(f"Missing keys after load: {len(missing_keys)}")
        print(f"Unexpected keys after load: {len(unexpected_keys)}")

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    if checkpoint_dir is None:
        print(f"Merged final adapter from: {adapter_path}")
    else:
        print(f"Merged checkpoint from: {checkpoint_dir}")
    print(f"Merged model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
