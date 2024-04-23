from transformers import get_linear_schedule_with_warmup, AdamW
import pandas as pd
import torch
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
print(device)



lyrics_train_df = pd.read_csv('./cleaned_train_lyrics.csv', index_col = False)
lyrics_test_df = pd.read_csv('./cleaned_test_lyrics.csv', index_col = False)

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')



from transformers import AutoTokenizer, AutoModelForCausalLM
# Load model directly

# tokenizer = AutoTokenizer.from_pretrained("D3STRON/multi-genre-alt")
# model = AutoModelForCausalLM.from_pretrained("D3STRON/multi-genre-alt")
tokenizer = AutoTokenizer.from_pretrained("D3STRON/multi-genre-medium")
model = AutoModelForCausalLM.from_pretrained("D3STRON/multi-genre-medium")
model.to(device)

tokenizer.add_special_tokens({
    'pad_token':'<|pad|>'
                             })


def calc_perp(model, tokenizer, test_loader):
    model.eval()
    nlls = []
    for lyric in tqdm(test_loader):
        lyric_tens = tokenizer(lyric, padding=True, truncation= True, return_tensors='pt')['input_ids'].to(device)
        target_tens = lyric_tens.clone()
        with torch.no_grad():
            outputs = model(lyric_tens, labels=target_tens)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).mean())
    print("Evaluations:", ppl.item())
    model.train()

class LyricsDataset(Dataset):
    def __init__(self, lyrics_df):
        super().__init__()
        self.lyric_list = []
        self.end_of_text_token = "<|endoflyric|>"
        
        for lyric, genre in tqdm(zip(lyrics_df['Lyric'],lyrics_df['genre'] ), total=len(lyrics_df['genre'])):
            lyric_str = f"LYRIC[{genre.lower()}]:{lyric}{self.end_of_text_token}"
            self.lyric_list.append(lyric_str)
        
    def __len__(self):
        return len(self.lyric_list)

    def __getitem__(self, item):
        return self.lyric_list[item]
    
train_data = LyricsDataset(lyrics_train_df)
test_data = LyricsDataset(lyrics_test_df)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 5e-6
total_training_steps = (len(train_data.lyric_list) // BATCH_SIZE) * EPOCHS
WARMUP_STEPS = int(0.1 * total_training_steps)
SAVE_STEPS = 100000
PRINT_STEPS = 100


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_training_steps)
proc_seq_count = 0


proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
steps = 0


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
proc_seq_count = 0


proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
steps = 0

model.train()
for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    for idx,lyric in enumerate(tqdm(train_loader)):
        lyric_tens = tokenizer(lyric, padding=True, return_tensors='pt', truncation= True)['input_ids'].to(device)
        output = model(lyric_tens, labels = lyric_tens)
        loss = output['loss']  / BATCH_SIZE
        loss.backward()
        sum_loss = sum_loss + output['loss'].detach().data
        
        proc_seq_count = proc_seq_count + 1
        steps += 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

        if steps >=SAVE_STEPS:
            steps = 0
            calc_perp(model, tokenizer, test_loader)
            model.push_to_hub("multi-genre-medium")
            tokenizer.push_to_hub("multi-genre-medium")