from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset
import re
import pandas as pd
import pdb

class SEMI_SSL_Dataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]
    

def get_time_complexity_pattern(found_patterns,sentence, time_complexity_expressions, idx):
    trigger = 0
    for pattern in time_complexity_expressions:
        matches = re.findall(pattern, sentence)

        if matches:
            found_patterns.append(pattern)
            trigger += 1
        
    if trigger == 0:
        found_patterns.append([idx,'Error'])
    
    return found_patterns


def main():
    if torch.cuda.is_available():
        device_idx = 0     
        device = torch.device("cuda", device_idx)
        print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
    else:
        print('\nNo GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # Initialize model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print("Model name => ", model_name,'\n\n')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    # Load data
    train_df = pd.read_csv('../data/problem_based_split/python_extended_data/train.csv')
    
    label_map = {1: 'constant',2: 'logn',3: 'linear',4: 'nlogn',5: 'quadratic',6: 'cubic',7: 'np'}
    train_df['label'] = train_df['label'].map(label_map)
    train_dataset = SEMI_SSL_Dataset(train_df['content'].to_list(), labels=train_df['label'].to_list())
    
    correct = 0
    not_correct = 0
    correct_idx = []
    not_correct_idx = []
    found_patterns = []
    time_complexity_expressions = [r"O\(1\)",r"O\(log n\)",r"O\(n\)", r"O\(n log n\)",r"O\(n\^2\)",r"O\(n\^3\)", r"O\(polynomial\)"]
    complexity_to_pattern = {r"O\(1\)": 'constant',r"O\(log n\)": 'logn',r"O\(n\)": 'linear',r"O\(n log n\)": 'nlogn',r"O\(n\^2\)": 'quadratic',r"O\(n\^3\)": 'cubic',r"O\(polynomial\)": 'np'}

    # Iterate over the dataset
    for idx, data in enumerate(train_dataset):
        messages = [
            {'role': 'user', 
            'content': (f"code : {data[0]}" 
            "Just tell me ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'np'] what you think the time complexity of this code is")}
        ]

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('\n')
        sentence = result[0]

        # Find time complexity patterns in the sentence
        found_patterns = get_time_complexity_pattern(found_patterns,sentence, time_complexity_expressions, idx)

        # Compare predicted complexity with true label
        predicted_complexity = complexity_to_pattern[found_patterns[idx]]
        
        true_label = train_df['label'][idx]
        if predicted_complexity == true_label:
            correct += 1
            correct_idx.append(idx)
        else:
            not_correct += 1
            not_correct_idx.append(idx)
            
        print("Model output => ", result)
        print("We use sentence(result[0]) => ", sentence)
        print(f"***Label => {true_label},  Predict => {predicted_complexity}***\n\n")


    print(f"Correct predictions: {correct}, indices: {correct_idx}")
    print(f"Incorrect predictions: {not_correct}, indices: {not_correct_idx}")
    print("Total list => ", [pattern.replace('\\', '') for pattern in found_patterns])

    
    print(f"Accuracy => {correct / len(train_dataset)}")

if __name__ == "__main__":
    main()
