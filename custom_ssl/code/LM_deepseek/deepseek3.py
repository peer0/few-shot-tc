from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset
import re
import pandas as pd
import pdb
from collections import Counter
from transformers import pipeline


class SEMI_SSL_Dataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]
    

def get_time_complexity_pattern(found_patterns, sentence, time_complexity_expressions, idx):
    trigger = 0
    for pattern in time_complexity_expressions:
        matches = re.findall(pattern, sentence)

        if matches:
            found_patterns.append(pattern)
            trigger += 1
        
    if trigger == 0:
        found_patterns.append('None')
    
    return found_patterns


def main():
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    from transformers import logging
    logging.set_verbosity_error() 
    
    if torch.cuda.is_available():
        device_idx = 0     
        device = torch.device("cuda", device_idx)
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # Initialize model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print("\n\nModel name => ", model_name,'\n\n')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

    
    # Load data
    test_df = pd.read_csv('../data/problem_based_split/python_extended_data/test.csv')
    train_df = pd.read_csv('./lm_train_data.csv')
    
    
    label_map = {1: 'constant',2: 'logn',3: 'linear',4: 'nlogn',5: 'quadratic',6: 'cubic',7: 'np'}
    map1 = {'constant': "O(1)", 'logn': "O(log n)", 'linear': "O(n)", 'nlogn': "O(n log n)", 'quadratic': "O(n^2)", 'cubic': "O(n^3)", 'np': "O(polynomial)"}

    
    test_df['label'] = test_df['label'].map(label_map)
   
    train_df['label'] = train_df['label'].map(label_map)
    train_df['label'] = train_df['label'].map(map1)
    
    
    test_dataset = SEMI_SSL_Dataset(test_df['content'].to_list(), labels=test_df['label'].to_list())
    train_dataset = SEMI_SSL_Dataset(train_df['content'].to_list(), labels=train_df['label'].to_list())
    
    item_counts = Counter(test_dataset[:][1])
    
    
    correct = 0
    not_correct = 0
    correct_idx = []
    not_correct_idx = []
    found_patterns = []
    time_complexity_expressions = [r"O\(1\)",r"O\(log n\)",r"O\(n\)", r"O\(n log n\)",r"O\(n\^2\)",r"O\(n\^3\)", r"O\(polynomial\)"]
    complexity_to_pattern = {r"O\(1\)": 'constant',r"O\(log n\)": 'logn',r"O\(n\)": 'linear',r"O\(n log n\)": 'nlogn',r"O\(n\^2\)": 'quadratic',r"O\(n\^3\)": 'cubic',r"O\(polynomial\)": 'np'}

    from tqdm import tqdm
    pbar = tqdm(total=len(test_dataset), desc="Training", position=0, leave=True)
    
    # Iterate over the dataset
    for idx, data in enumerate(test_dataset):
        messages = [
            {f'role': 'user', 
            f'content': 
            #"Predict code time complexity example:"
            #f"Code: {train_dataset[:][0]}, Label: {train_dataset[:][1]}\n"
            #f"Can you tell me the time complexity of the code in the following example?\n"
            f"Can you tell me the time complexity of the code based on "
            f"\n1. O(1) \n2. O(log n) \n3. O(n) \n4. O(n log n) \n5. O(n^2) \n6. O(n^3) \n7. O(np)?\n"
            f"{data[0]}" }   
            #f"Can you tell me the time complexity of the code?\n"
            #f"Just tell me ['O(1)','O(log n)','O(n)', 'O(n log n)','O(n^2)','O(n^3)', 'O(polynomial)'] what you think the time complexity of this code is"}
        ]
        
       
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).split('\n')
        sentence = result[0]

        # Find time complexity patterns in the sentence
        found_patterns = get_time_complexity_pattern(found_patterns,sentence, time_complexity_expressions, idx)
        pdb.set_trace()
        # Compare predicted complexity with true label
        if found_patterns[idx] in  time_complexity_expressions : 
            predicted_complexity = complexity_to_pattern[found_patterns[idx]]
        
        else :
            predicted_complexity = None
        
        
        true_label = test_df['label'][idx]
        if predicted_complexity == true_label:
            correct += 1
            correct_idx.append(idx)
        else:
            not_correct += 1
            not_correct_idx.append(idx)
            
        print(f"Code-data_idx = {idx}, Model output => ", result)
        print("We use sentence(result[0]) => ", sentence)
        print(f"***Label => {true_label},  Predict => {predicted_complexity}***\n\n")
        
        
        pbar.update(1)
        
        #if idx == 10:
        #    break
        
    pbar.close()    

    total_list = [pattern.replace("\\", '') for pattern in found_patterns]
    print(f"Correct predictions: {correct}, indices: {correct_idx}")
    print(f"Incorrect predictions: {not_correct}, indices: {not_correct_idx}")
    print("Total list => ",  total_list)

    correct_elements = [complexity_to_pattern[found_patterns[idx]] for idx in correct_idx]
    correct_elements = Counter(correct_elements)
   
   
    print(f"\nAccuracy => {correct / len(test_dataset)}")
    print("Total precdict label => ", total_list)
    
    for item, count in item_counts.items():
        print(f"{item}accuracy => {correct_elements[item]/count}")
    
    
    
if __name__ == "__main__":
    main()
