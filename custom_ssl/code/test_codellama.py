from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset
import re
import pandas as pd
from collections import Counter
import pdb
from transformers import pipeline

# 데이터셋 클래스 정의
class SEMI_SSL_Dataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

# 시간 복잡도 패턴 찾는 함수 정의
def get_time_complexity_pattern(found_patterns, sentence, time_complexity_expressions):
    trigger = 0
    found = []
    for pattern in time_complexity_expressions:
        matches = re.findall(pattern, sentence)

        if matches:
            found_patterns.append(pattern)
            found.append(pattern)
            trigger += 1
        
    if trigger == 0:
        found_patterns.append(None)
        found.append(None)
        
    return found, found_patterns

def main(model_name):
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    from transformers import logging
    logging.set_verbosity_error() 
    
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        device_idx = 0     
        device = torch.device("cuda", device_idx)
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    # 모델과 토크나이저 초기화
    model_name = model_name
    print("\n\nModel name => ", model_name,'\n\n')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

    # 데이터 로드
    test_df = pd.read_csv('../data/problem_based_split/python_extended_data/test.csv')

    
    # 레이블 매핑
    label_map = {1: 'constant',2: 'logn',3: 'linear',4: 'nlogn',5: 'quadratic',6: 'cubic',7: 'np'}
    #label_map = {7: 'np'}
    test_df['label'] = test_df['label'].map(label_map)
    
    print('\nData_label => ', Counter(test_df.label))
    
    # 데이터셋 생성
    test_dataset = SEMI_SSL_Dataset(test_df['content'].to_list(), labels=test_df['label'].to_list())
    
    # 레이블 개수 계산
    item_counts = Counter(test_dataset[:][1])
    
    correct = 0
    not_correct = 0
    correct_idx = []
    not_correct_idx = []
    found_patterns = []
    time_complexity_expressions = [r"O\(1\)",r"O\(log n\)",r"O\(n\)", r"O\(n log n\)",r"O\(n\^2\)",r"O\(n\^3\)", r"O\(2\^n\)"]
    complexity_to_pattern = {r"O\(1\)": 'constant',r"O\(log n\)": 'logn',r"O\(n\)": 'linear',r"O\(n log n\)": 'nlogn',r"O\(n\^2\)": 'quadratic',r"O\(n\^3\)": 'cubic',r"O\(2\^n\)": 'np'}
    pattern_to_complexity = {'constant': r"O\(1\)", 'logn': r"O\(log n\)", 'linear': r"O\(n\)", 'nlogn': r"O\(n log n\)", 'quadratic': r"O\(n\^2\)", 'cubic': r"O\(n\^3\)", 'np': r"O\(2\^n\)"}
    
    from tqdm import tqdm
    pbar = tqdm(total=len(test_dataset), desc="Training", position=0, leave=True)
    code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    # 데이터셋 순회
    for idx, data in enumerate(test_dataset):
        messages = [
            {f'role': 'user', 
            f'content': 
            f"{data[0]}" 
            f"Can you tell me the time complexity of the code based on "
            f"\n1. O(1) \n2. O(log n) \n3. O(n) \n4. O(n log n) \n5. O(n^2) \n6. O(n^3) \n7. O(2^n)?\n"
            f"Say something like, “**The time complexity of this code is (time complexity of code)."    }]
       
       
        input_string = messages
        generated_code = code_generator(input_string, max_length=200)[0]['generated_text']
        sentence = generated_code

        # 시간 복잡도 패턴 찾기
        output_list, found_patterns = get_time_complexity_pattern(found_patterns,sentence, time_complexity_expressions)

        
        # 예측된 복잡도와 실제 레이블 비교
        if output_list[0] in  time_complexity_expressions : 
            predicted_complexity = complexity_to_pattern[output_list[0]]
        
        else :
                predicted_complexity = None
            
        
        true_label = test_df['label'][idx]
        if predicted_complexity == true_label:
            correct += 1
            correct_idx.append(idx)
        else:
            not_correct += 1
            not_correct_idx.append(idx)
        
          
        #print(f"Code-data_idx = {idx} \nModel output => ", result)
        print("We use sentence(result[0]) => ", sentence)
        print(f"***Label = {true_label} | Predict = {predicted_complexity}*** \ncorrect ==> {true_label == predicted_complexity}")
        print(f"Lable symobol => { pattern_to_complexity[true_label]} | Output_list => {output_list}\n\n")  
        
        pbar.update(1)
        
 
    pbar.close()    

    
    print(f"Correct predictions: {correct} \nindices: {correct_idx}")
    print(f"Incorrect predictions: {not_correct} \nindices: {not_correct_idx}")

    correct_elements = [complexity_to_pattern[found_patterns[idx]] for idx in correct_idx]
    correct_elements = Counter(correct_elements)
   
    print(f"\nAccuracy => {correct / len(test_dataset)}")
    
    for item, count in item_counts.items():
        print(f"{item}-accuracy => {correct_elements[item]/count}")
   
   
   
    
if __name__ == "__main__":
    main("codellama/CodeLlama-7b-Instruct-hf")
    
    