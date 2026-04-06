# To merge the lora adapter weights into the base model tinyllama model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Replace these paths with the actual paths on your system
base_model_path = "/scratch/user/varshagoalla/huggingface/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
adapter_path = "/scratch/user/varshagoalla/self-rag/retrieval_lm/output/tinyllama_selfrag_sft"
output_path = "/scratch/user/varshagoalla/self-rag/retrieval_lm/output/tinyllama_selfrag_sft_merged"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto")
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"Merged model saved to: {output_path}")
