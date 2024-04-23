import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import datasets 
# from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


lyrics_test_df = pd.read_csv('./cleaned_test_lyrics.csv')


class LyricsDataset(Dataset):
    def __init__(self, lyrics_df):
        super().__init__()


        self.lyric_list = []
        self.end_of_text_token = "<|endoflyric|>"
        
        for lyric, genre in tqdm(zip(lyrics_df['Lyric'],lyrics_df['genre'] ), total=len(lyrics_df['genre'])):
            lyric_str = f"LYRIC[{genre.lower()}]:{lyric}{self.end_of_text_token}"
#             lyric_str = f"LYRIC:{lyric}{self.end_of_text_token}"
            self.lyric_list.append(lyric_str)
        
    def __len__(self):
        return len(self.lyric_list)

    def __getitem__(self, item):
        return self.lyric_list[item]
    

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("D3STRON/multi_genre_music_generator")
# model = AutoModelForCausalLM.from_pretrained("D3STRON/multi_genre_music_generator")

# tokenizer.add_special_tokens({
#     'pad_token':'<|pad|>'
#                              })
model.to(device)

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

for genre in lyrics_test_df['genre'].unique():
    # train_data = LyricsDataset(lyrics_train_df)
    print(f'For {genre}:')
    test_data = LyricsDataset(lyrics_test_df[lyrics_test_df['genre'] == genre])

    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    evaluate(model, tokenizer, test_loader)