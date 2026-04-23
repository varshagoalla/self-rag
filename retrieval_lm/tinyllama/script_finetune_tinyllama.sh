export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
OUTPUT_DIR=output/tinyllama_selfrag_sft_r16_a16_e3
TRAIN_FILE=train.jsonl
NUM_GPUS=4

accelerate launch \
    --mixed_precision bf16 \
    --num_processes $NUM_GPUS \
    finetune.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --use_special_tokens \
    --max_seq_length 768 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --checkpointing_steps epoch \
    --logging_steps 10 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to tensorboard
