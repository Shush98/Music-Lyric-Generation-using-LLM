import pandas as pd
import transformers
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM , AutoTokenizer
from transformers import pipeline, set_seed
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
#from lightning.pytorch.loggers import TensorBoardLogger

#Dataset
from datasets import load_dataset

#PEFT
from peft import LoraConfig
from peft import PeftConfig
from peft import PeftModel
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
import torch

import warnings
warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()


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
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
device = 'cuda'


config_name = "D3STRON/LLama_music_generator"
config = PeftConfig.from_pretrained(config_name)
model.load_adapter(config_name)


from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

lyrics_test_df = pd.read_csv('./keyword_lyrics_test.csv')

class LyricsDataset(Dataset):
    def __init__(self, lyrics_df):
        super().__init__()


        self.lyric_list = []

        for lyric, genre, keyword in tqdm(zip(lyrics_df['Lyric'], lyrics_df['genre'], lyrics_df['keywords']), total=len(lyrics_df['genre'])):
            lyric_str = f"### USER: Generate [{genre}] song lyrics having keywords: {keyword}\n### ASSISTANT: {lyric}"
#             lyric_str = f"LYRIC:{lyric}{self.end_of_text_token}"
            self.lyric_list.append(lyric_str)

    def __len__(self):
        return len(self.lyric_list)

    def __getitem__(self, item):
        return self.lyric_list[item]

test_data = LyricsDataset(lyrics_test_df)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)


def evaluate(model, tokenizer, test_loader):
    model.eval()
    nlls = []
    EVAL_STEPS = 1000
    steps = 0
    for lyric in tqdm(test_loader):
        lyric_tens = tokenizer(lyric, padding=True, truncation= True, return_tensors='pt')['input_ids'].to(device)
        target_tens = lyric_tens.clone()
        with torch.no_grad():
            outputs = model(lyric_tens, labels=target_tens)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
        if EVAL_STEPS == steps:
            steps = 0
            ppl = torch.exp(torch.stack(nlls).mean())
            print("Evaluations:", ppl.item())
        steps += 1

test_data = LyricsDataset(lyrics_test_df[lyrics_test_df['genre'] == 'rap'])
# train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
evaluate(model, tokenizer, test_loader)