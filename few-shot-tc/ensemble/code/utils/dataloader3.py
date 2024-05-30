#initial labeled data + forwhile aug (인위적인셋팅)
import os
import sys
import random
import torch
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import csv

class SEMIDataset(Dataset):
    def __init__(self, sents, sents_aug1, sents_aug2, labels=None):
        self.sents = sents
        self.sents_aug1 = sents_aug1
        self.sents_aug2 = sents_aug2
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.sents_aug1[idx], self.sents_aug2[idx], self.labels[idx]

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
            idx = self.sents.index(new_sent)
            self.labels[idx] = new_label
        else:
            self.sents.append(new_sent)
            self.labels.append(new_label)
            


class SEMINoAugDataset(Dataset):
    def __init__(self, sents, labels=None):
        self.sents = sents
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sents[idx], self.labels[idx]

class MyCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sents, sents_aug1, sents_aug2 = [], [], []
        labels = []
        for sample in batch:
            if len(sample) == 2:
                sents.append(sample[0])
                labels.append(sample[1])
                sents_aug1 = None
                sents_aug2 = None
            elif len(sample) == 4:
                sents.append(sample[0])
                sents_aug1.append(sample[1])
                sents_aug2.append(sample[2])
                labels.append(sample[3])
    
        tokenized = self.tokenizer(sents, padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
        labels = torch.LongTensor(labels) - 1
        if sents_aug1 is not None:
            # further add stochastic synoym replacement augmentation
            sents_aug1 = [naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment(sent)[0] for sent in sents_aug1]
            tokenized_aug1 = self.tokenizer(sents_aug1, padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
        else:
            # add stochastic synoym replacement augmentation
            sents_aug1 = [naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment(sent)[0] for sent in sents]
            tokenized_aug1 = self.tokenizer(sents_aug1, padding=True, truncation='longest_first', max_length=512, return_tensors='pt')            
        if sents_aug2 is not None: 
            # further add stochastic synoym replacement augmentation
            sents_aug2 = [naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment(sent)[0] for sent in sents_aug2]
            tokenized_aug2 = self.tokenizer(sents_aug2, padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
        else:
            tokenized_aug2 = None
        # return tokenized, tokenized_aug1, 
            tokenized_aug2, labels
        return {'x': tokenized, 'x_w': tokenized_aug1,'x_s': tokenized_aug2, 'label': labels}
    
    
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

    
        tokenized = self.tokenizer(sents, padding=True, truncation='longest_first', max_length=512, return_tensors='pt')
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

    def __iter__(self):
        # Calculate the number of iterations
        num_iterations = len(self.dataset) // self.batch_size

        for _ in range(num_iterations):
            # Shuffle class indices
            class_order = np.random.permutation(self.num_classes)
            # Select one sample from each class for each batch
            for i in class_order:
                yield self.class_indices[i][0]
                self.class_indices[i] = np.roll(self.class_indices[i], -1)  # Move selected index to the end

    def __len__(self):
        return len(self.dataset)

def train_split(labels, n_labeled_per_class, unlabeled_per_class=None): #unlabeled_per_class를 이용해서 ul_data를 설정 가능 현재는 전체.
    """Split the dataset into labeled and unlabeled subsets.
    Args:
        labels: labels of the training data
        n_labeled_per_class: number of labeled examples per class
        unlabeled_per_class: number of unlabeled examples per class
        Returns:
            train_labeled_idxs: list of labeled example indices
            train_unlabeled_idxs: list of unlabeled example indices
    """
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


def manually_train_split(aug_df, train_df, n_labeled_per_class):
    train_labeled_idxs = []
    for label in train_df['label'].unique():
        # 각 레이블에 해당하는 데이터를 가져옵니다.
        label_df = train_df[train_df['label'] == label]
        # aug_df에 있는 데이터만 선택합니다.
        label_df = label_df[label_df['idx'].isin(aug_df['idx'])]
        # 각 클래스에서 n_labeled_per_class 만큼의 데이터를 무작위로 선택합니다.
        label_idxs = label_df.sample(n_labeled_per_class, replace=False).index.tolist()
        train_labeled_idxs.extend(label_idxs)

    # train_unlabeled_idxs는 train_labeled_idxs에 사용되지 않은 데이터를 포함합니다.
    train_unlabeled_idxs = [idx for idx in train_df.index if idx not in train_labeled_idxs]

    return train_labeled_idxs, train_unlabeled_idxs


def jsonl_to_csv(jsonl_file, csv_file):
    with open(jsonl_file, 'r') as f:
        data = f.readlines()
    
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['src', 'forwhile', 'index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for line in data:
            json_data = json.loads(line)
            writer.writerow({'src': json_data['src'], 'forwhile': json_data['forwhile'], 'index': json_data['index']})


def get_dataloader(data_path, dataset, n_labeled_per_class, bs, load_mode='semi_SSL', token = None):
    if token == "microsoft/codebert-base":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    elif token == "Salesforce/codet5p-110m-embedding":
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-110m-embedding", trust_remote_code=True)
    elif token == "microsoft/unixcoder-base":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    elif token == "microsoft/graphcodebert-base":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    elif token == "codesage/codesage-base":
        tokenizer = AutoTokenizer.from_pretrained("codesage/codesage-base")


    train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
    dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
 
    aug_path = f'../data/aug/{dataset}'
    jsonl_file = f'{aug_path}/cc_{dataset}_looptranslation.jsonl'
    csv_file = f'{aug_path}/cc_{dataset}_looptranslation.csv'
    jsonl_to_csv(jsonl_file, csv_file)
    aug_df = pd.read_csv(f'{aug_path}/cc_{dataset}_looptranslation.csv')
    # 'forwhile' 열을 'content'로, 'index'를 'idx'로 변경
    aug_df = aug_df.rename(columns={'forwhile': 'content', 'index': 'idx'})
    aug_df = aug_df[~aug_df['content'].str.contains('ERROR')] #Error가 포함되어 있는 데이터는 제거
    
    # l_train_df = train_df[train_df['idx'].isin(aug_df['idx'])] 
    # l_labels = list(l_train_df["label"])
    labels = list(train_df["label"])
    num_class = len(set(labels))
    
    #train_labeled_idxs는 aug_df에 있는 데이터만 선택
    
    train_labeled_idxs, train_unlabeled_idxs = manually_train_split(aug_df, train_df, n_labeled_per_class)
    
    train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
    
    print("initial labeled dataset개수:", len(train_l_df))
    print("initial labeled data index별 개수:",train_l_df['idx'].value_counts().to_dict())
    
    # train_l_df의 'idx' 값이 aug_df에도 있는 행만 선택(aug_df에 없으면 for나 while이 없는 코드임)
    aug_df = aug_df[aug_df['idx'].isin(train_l_df['idx'])]
    concat_df = pd.concat([train_l_df, aug_df], ignore_index=True)
    
    # 'idx'가 같은 행에서 'Label' 값을 train_l_df의 값으로 업데이트(aug 데이터엔 label값이 비어있어서 채워줌)
    for idx in concat_df['idx'].unique():
        if idx in train_l_df['idx'].values:
            mask = (concat_df['idx'] == idx)
            concat_df.loc[mask, 'label'] = train_l_df.loc[train_l_df['idx'] == idx, 'label'].values[0]

    print("initial labeled dataset+aug개수:", len(concat_df))
    print("initial labeled data+aug index별 개수:",concat_df['idx'].value_counts().to_dict())
    concat_df['label'] = concat_df['label'].astype(int)
    train_l_df = concat_df
    
    
    # # check statistics info
    print('n_labeled_per_class: ', n_labeled_per_class)
    print('train_df samples: %d' % (train_df.shape[0]))
    print('train_labeled_df samples: %d' % (train_l_df.shape[0]))
    print('train_unlabeled_df samples: %d' % (train_u_df.shape[0]))

    
    if load_mode == 'semi_SSL':
        train_dataset_l = SEMI_SSL_Dataset(train_l_df['content'].to_list(), labels=train_l_df['label'].to_list())
        train_dataset_u = SEMI_SSL_Dataset(train_u_df['content'].to_list(), labels=train_u_df['label'].to_list())
        
        train_loader_u = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))
    
    train_sampler = BalancedBatchSampler(train_dataset_l,bs)
    train_loader_l = DataLoader(dataset=train_dataset_l, batch_size=bs, sampler=train_sampler,collate_fn=MyCollator_SSL(tokenizer))
    
    dev_dataset = SEMINoAugDataset(dev_df['content'].to_list(), labels=dev_df['label'].to_list())
    test_dataset = SEMINoAugDataset(test_df['content'].to_list(), labels=test_df['label'].to_list())

    dev_loader = DataLoader(dataset=dev_dataset, batch_size= 1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))
    test_loader = DataLoader(dataset=test_dataset, batch_size= 1, shuffle=False, collate_fn=MyCollator_SSL(tokenizer))

    return train_loader_l, train_loader_u, dev_loader, test_loader, num_class, train_dataset_l, train_dataset_u

