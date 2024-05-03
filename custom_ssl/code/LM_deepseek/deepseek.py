from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import pdb

class SEMINoAugDataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

class SEMI_SSL_Dataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]
    


    def add_data(self, new_sent, new_label):
        if new_sent in self.sents:
            # 해당 데이터가 이미 존재하는 경우 레이블을 업데이트합니다.
            #print("**동일 데이터 업데이트")
            idx = self.sents.index(new_sent)
            self.labels[idx] = new_label
        else:
            # 새로운 데이터인 경우 데이터와 레이블을 추가합니다.
            #print("*추가 데이터 업데이트")
            self.sents.append(new_sent)
            self.labels.append(new_label)
  
class MyCollator_SSL(object): # 추가 SSL
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sents = []
        labels = []
        for sample in batch:
            if len(sample) == 2:
                sents.append(sample[0])
                labels.append(sample[1])

        
        sents2 = f"code : {sents[0]}. What do you think the time complexity of this code is? Please tell me based on this [1: 'constant', 2: 'logn', 3: 'linear', 4: 'nlogn', 5: 'quadratic', 6: 'cubic', 7: 'np'] number."
        tokenized = self.tokenizer(sents2, padding=True, truncation='longest_first', max_length= 512, return_tensors='pt').to(device)
        labels = torch.LongTensor(labels) - 1
                     

        return {'x': tokenized,'label': labels}

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_classes = len(np.unique(dataset.labels))
        # Adjust for 1-indexed class labels
        self.class_indices = [np.where(np.array(dataset.labels) == i+1)[0] for i in range(self.num_classes)]
        self.batch_size = batch_size  # Set batch size equal to the number of classes

class myclass:
    def __init__(self, model, data_path):
        self.model = model
        self.data_path = data_path
        
        if torch.cuda.is_available():
            device_idx = 0     
            device = torch.device("cuda", device_idx)
            print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
        else:
            print('\nNo GPU available, using the CPU instead.')
            device = torch.device("cpu")
            
        self.device = device
            
        if self.data_path == 'java':
            data_path = '../data/problem_based_split/java_extended_data' 
            
        elif self.data_path == 'python':
            data_path = '../data/problem_based_split/python_extended_data' 
        
        elif self.data_path == 'corcod':
            data_path = '../data/problem_based_split/corcod.index'
        
        self.train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
        self.dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
        
        
        self.labels = list(self.train_df["label"])
        self.num_class = len(set(self.labels))
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).to(device)

    def get_value(self):
        return self.model, self.tokenizer, self.train_df, self.dev_df, self.test_df, self.labels, self.num_class, self.device
    
def train_split(labels, n_labeled_per_class, unlabeled_per_class=None): #unlabeled_per_class를 이용해서 ul_data를 설정 가능 현재는 전체.
    labels = np.array(labels)
    all_classes = set(labels)

    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in all_classes:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        if unlabeled_per_class:
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:n_labeled_per_class+unlabeled_per_class])
        else: 
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    
    
    return train_labeled_idxs, train_unlabeled_idxs

def test(data, device):
    for i in data:
        pdb.set_trace()
        outputs = model.generate(**i['x'], max_length= 512)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    
    
    
    
    
    
    
LM = myclass("deepseek-ai/deepseek-coder-6.7b-base", 'python')
    
model, tokenizer, train_df, dev_df, test_df, labels, num_class, device = LM.get_value()

train_labeled_idxs, train_unlabeled_idxs = train_split(labels, 1)
train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)

train_dataset_l = SEMI_SSL_Dataset(train_l_df['content'].to_list(), labels=train_l_df['label'].to_list())
train_dataset_u = SEMI_SSL_Dataset(train_u_df['content'].to_list(), labels=train_u_df['label'].to_list())
dev_dataset = SEMINoAugDataset(dev_df['content'].to_list(), labels=dev_df['label'].to_list())
test_dataset = SEMINoAugDataset(test_df['content'].to_list(), labels=test_df['label'].to_list())


train_loader_l = DataLoader(dataset= train_dataset_l, batch_size= 1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))
train_loader_u = DataLoader(dataset= train_dataset_u , batch_size= 1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))
dev_loader = DataLoader(dataset=dev_dataset, batch_size= 1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))
test_loader = DataLoader(dataset=test_dataset, batch_size= 1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))
    



test(train_loader_l, device)