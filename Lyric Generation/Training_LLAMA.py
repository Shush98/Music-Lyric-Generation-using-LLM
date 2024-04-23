from datasets import Dataset, DatasetDict, load_dataset

# temp_train_df = keyword_train_df[['Lyric', 'genre', 'keywords']]
# temp_test_df = keyword_test_df[['Lyric', 'genre', 'keywords']]

# train_dataset = Dataset.from_pandas(temp_train_df)
# test_dataset = Dataset.from_pandas(temp_test_df)
# dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
dataset = load_dataset("D3STRON/music_lyrics_5k")

import transformers
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM , AutoTokenizer
from transformers import pipeline, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
#from lightning.pytorch.loggers import TensorBoardLogger

#Dataset
from datasets import load_dataset

#PEFT
from peft import LoraConfig
from peft import PeftConfig
from peft import PeftModel, PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
import torch

import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param}")


from transformers import AutoTokenizer

# model_name = "microsoft/phi-2"
# tokenizer_name = "microsoft/phi-2"

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer_name = "meta-llama/Llama-2-7b-hf"

# model_name = "D3STRON/multi_genre_music_generator"
# tokenizer_name = "D3STRON/multi_genre_music_generator"

#Bits and Bytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #4bit quantizaition - load_in_4bit is used to load models in 4-bit quantization 
    bnb_4bit_use_double_quant=True, #nested quantization technique for even greater memory efficiency without sacrificing performance. This technique has proven beneficial, especially when fine-tuning large models
    bnb_4bit_quant_type="nf4", #quantization type used is 4 bit Normal Float Quantization- The NF4 data type is designed for weights initialized using a normal distribution
    bnb_4bit_compute_dtype=torch.bfloat16, #modify the data type used during computation. This can result in speed improvements. 
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map="auto",
                                                      trust_remote_code=True, 
                                                      quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

config_name = "D3STRON/LLAMA_lyric_generator"
# config = PeftConfig.from_pretrained(config_name)
tokenizer = AutoTokenizer.from_pretrained(config_name)


# for name, param in model.named_parameters():
#   if "lora" in name:  # Adjust the keyword based on your LoRA layer naming convention
#     param.requires_grad = True

tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens({'pad_token': '<PAD>'})

# Enable gradient checkpointing for the model. Gradient checkpointing is a technique used to reduce the memory consumption during the backward pas. Instead of storing all intermediate activations in the forward pass (which is what's typically done to compute gradients in the backward pass), gradient checkpointing stores only a subset of them
model.gradient_checkpointing_enable() 

# Prepare the model for k-bit training . Applies some preprocessing to the model to prepare it for training.
model = prepare_model_for_kbit_training(model)


#If targeting all linear layers
# target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

    
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules = target_modules,
#     bias="none",
#     lora_dropout=0.1,
#     task_type="CAUSAL_LM",
# )

model.load_adapter(config_name)


# Print the number of trainable parameters in the model
print_trainable_parameters(model)


from transformers import TrainingArguments

#Training Parameters
batch_size = 4
output_dir = f"LLama_music_generator"
per_device_train_batch_size = batch_size
gradient_accumulation_steps = 4
optim = 'adamw_hf' #"paged_adamw_32bit" #"paged_adamw_8bit"
save_steps = 10
save_total_limit=3
logging_steps = 10
learning_rate = 1e-5
max_grad_norm = 0.3
warmup_ratio = 0.04
lr_scheduler_type = 'constant_with_warmup'#"cosine"
epochs=2

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    num_train_epochs=epochs, 
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    save_strategy='steps',
    max_grad_norm=max_grad_norm,
    # max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
    push_to_hub=True,
)


def formatting_func(example):
    text = f"### USER: Generate [{example['genre']}] song lyrics having keywords: {example['keywords']}\n### ASSISTANT: {example['Lyric']}"
    return text


trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    packing=True,
    #dataset_text_field="id",
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_func,
)

trainer.train()