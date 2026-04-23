#!/usr/bin/env python

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import DPOConfig, DPOTrainer


class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps=100):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            torch.cuda.empty_cache()
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--empty-cache-steps", type=int, default=100)
    args = parser.parse_args()

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["eval"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"torch_dtype": compute_dtype}
    if args.load_in_4bit:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if torch.cuda.is_available() and local_rank >= 0:
            model_kwargs["device_map"] = {"": local_rank}
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "config"):
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if args.eval_file else "no",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        peft_config=peft_config,
    )
    trainer.add_callback(EmptyCacheCallback(args.empty_cache_steps))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
