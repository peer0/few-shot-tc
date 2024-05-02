from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import numpy as np
import os
import pdb

class SEMI_SSL_Dataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]
    
def setting(name, data_path):
        if data_path == 'java':
            data_path = '../data/problem_based_split/java_extended_data' 
            
        elif data_path == 'python':
            data_path = '../data/problem_based_split/python_extended_data' 
        
        elif data_path == 'corcod':
            data_path = '../data/problem_based_split/corcod.index'
        
        train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
        #dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
        #test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
        
        
        #labels = list(train_df["label"])
        #num_class = len(set(labels))
        
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).to(device)
        
        
        return model, tokenizer, train_df

if torch.cuda.is_available():
        device_idx = 0     
        device = torch.device("cuda", device_idx)
        print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
else:
    print('\nNo GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
correct = 0
correct_idx = []
not_correct = 0 
not_correct_idx = []
found_patterns = []

model, tokenizer, train_df = setting("deepseek-ai/deepseek-coder-1.3b-instruct", 'python')   
#time_complexity_expressions = [ "O(1)","O(log n)","O(n)","O(n log n)","O(n^2)","O(n^3)","O(polynomial)"]
time_complexity_expressions = [r"O\(1\)",r"O\(log n\)",r"O\(n\)", r"O\(n log n\)",r"O\(n\^2\)",r"O\(n\^3\)", r"O\(polynomial\)"]

patterns = [r'\bconstant\b', r'\blogn\b', r'\blinear\b', r'\bnlogn\b', r'\bquadratic\b', r'\bcubic\b', r'\bnp\b']
#time_complexity_mapping = {r'\bconstant\b': 'constant', r'\blogn\b': 'logn',r'\blinear\b': 'linear',r'\bnlogn\b': 'nlogn',r'\bquadratic\b': 'quadratic',r'\bcubic\b': 'cubic',r'\bnp\b': 'np'}

#complexity_to_pattern = {'O(1)': r'\bconstant\b','O(log n)': r'\blogn\b','O(n)': r'\blinear\b','O(n log n)': r'\bnlogn\b','O(n^2)': r'\bquadratic\b','O(n^3)': r'\bcubic\b','O(polynomial)': r'\bnp\b'}
#complexity_to_pattern = {r"O\(1\)": r'\bconstant\b',r"O\(log n\)": r'\blogn\b',r"O\(n\)": r'\blinear\b',
#                         r"O\(n log n\)": r'\bnlogn\b',r"O\(n\^2\)": r'\bquadratic\b',r"O\(n\^3\)": r'\bcubic\b',
#                         r"O\(polynomial\)": r'\bnp\b'}

complexity_to_pattern = {
    r"O\(1\)": 'constant',
    r"O\(log n\)": 'logn',
    r"O\(n\)": 'linear',
    r"O\(n log n\)": 'nlogn',
    r"O\(n\^2\)": 'quadratic',
    r"O\(n\^3\)": 'cubic',
    r"O\(polynomial\)": 'np'
}


label_map = {1: 'constant',2: 'logn',3: 'linear',4: 'nlogn',5: 'quadratic',6: 'cubic',7: 'np'}

train_df['label'] = train_df['label'].map(label_map)

train_dataset = SEMI_SSL_Dataset(train_df['content'].to_list(), labels=train_df['label'].to_list())


for idx, data in enumerate(train_dataset):
    messages=[
        { 'role': 'user', 
        'content': (f"code : {data[0]}" 
                    "Just tell me ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'np'] what you think the time complexity of this code is")}
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    # tokenizer.eos_token_id is the id of <|EOT|> token
    outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('\n')
    sentence = result[0]

    # 각 패턴에 대해 반복하여 문장에서 패턴을 찾음
    for pattern in time_complexity_expressions :
        # 현재 패턴을 찾아서 matches 리스트에 저장
        matches = re.findall(pattern, sentence)
        # 만약 현재 패턴이 문장에 있으면 found_patterns 리스트에 추가
        
        if matches:
            found_patterns.append(pattern)
        else:
             found_patterns.append('error')
       
   
   
       
    #mapped_patterns = [time_complexity_mapping[pattern] for pattern in found_patterns]   
    print(f'\n\n\nidx => {idx}')
    print(result)    
    pdb.set_trace()
    print(f"***predict => {complexity_to_pattern[found_patterns[idx]]}, label => {train_df['label'][idx]}***")
    

    if complexity_to_pattern[found_patterns[idx]] == train_df['label'][idx]:
        correct +=1
        correct_idx.append(idx)
        
    else:
        not_correct += 1
        not_correct_idx.append(idx)
    
    
    if idx == 2:
        break
    
print(f"correct => {correct}, idx => {correct_idx}")
print(f"not_correct => {not_correct}, idx => {not_correct_idx}")
print(f"Total_list => {found_patterns}")
