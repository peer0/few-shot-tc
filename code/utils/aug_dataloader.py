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
    
    
class MyCollator_SSL(object):
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


#train split for natural
def train_split_v1(aug_error_df, train_df, n_labeled_per_class):
    error_idxs = aug_error_df['idx'].tolist()

    train_labeled_idxs = []
    for label in train_df['label'].unique():
        label_df = train_df[train_df['label'] == label]
        label_df = label_df[~label_df.isin(error_idxs)]
        label_idxs = label_df.sample(n_labeled_per_class, replace=False).index.tolist()
        train_labeled_idxs.extend(label_idxs)

    train_unlabeled_idxs = [idx for idx in train_df.index if idx not in train_labeled_idxs]

    return train_labeled_idxs, train_unlabeled_idxs


#train split for artificial
def train_split_v2(aug_df, train_df, n_labeled_per_class):
    train_labeled_idxs = []
    for label in train_df['label'].unique():
        label_df = train_df[train_df['label'] == label]
        #label_df = label_df[label_df.isin(aug_df['idx'])]
        print(aug_df)
        print(label_df)
        label_df = label_df[(label_df['idx']+1).isin(aug_df['idx'])]
        label_idxs = label_df.sample(n_labeled_per_class, replace=False).index.tolist()
        train_labeled_idxs.extend(label_idxs)

    train_unlabeled_idxs = [idx for idx in train_df.index if idx not in train_labeled_idxs]

    return train_labeled_idxs, train_unlabeled_idxs


#train split for natural (loop translation + backtranlation)
def train_split_v3(forwhile_error_df, backtrans_error_df, train_df, n_labeled_per_class):
    error_idxs = set(forwhile_error_df['idx'].tolist()).union(set(backtrans_error_df['idx'].tolist()))

    train_labeled_idxs = []
    
    for label in train_df['label'].unique():
        label_df = train_df[train_df['label'] == label]
        label_df = label_df[~label_df.isin(error_idxs)]
        label_idxs = label_df.sample(n_labeled_per_class, replace=False).index.tolist()
        train_labeled_idxs.extend(label_idxs)

    train_unlabeled_idxs = [idx for idx in train_df.index if idx not in train_labeled_idxs]

    return train_labeled_idxs, train_unlabeled_idxs


#train split for artificial (loop translation + backtranlation)
def train_split_v4(forwhile_df, backtrans_df, train_df, n_labeled_per_class):
    train_labeled_idxs = []
    common_idx = set(forwhile_df['idx']).intersection(set(backtrans_df['idx']))
    for label in train_df['label'].unique():
        label_df = train_df[train_df['label'] == label]
        label_df = label_df[(label_df['idx']).isin(common_idx)]
        #label_df = label_df[label_df.isin(common_idx)]
        label_idxs = label_df.sample(n_labeled_per_class, replace=False).index.tolist()
        train_labeled_idxs.extend(label_idxs)

    train_unlabeled_idxs = [idx for idx in train_df.index if idx not in train_labeled_idxs]

    return train_labeled_idxs, train_unlabeled_idxs


def jsonl_to_csv(jsonl_file, csv_file, aug):
    with open(jsonl_file, 'r') as f:
        data = f.readlines()
    
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['src', aug, 'index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in data:
            json_data = json.loads(line)
            writer.writerow({'src': json_data['src'], aug: json_data[aug], 'index': json_data['index']})


#natural
def get_dataloader_v1(data_path, dataset, n_labeled_per_class, bs, load_mode='semi_SSL',aug=None,  token = None):

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
    if dataset == 'corcod':
        jsonl_file = f'{aug_path}/cc_{dataset}_java_{aug}.jsonl'
        csv_file = f'{aug_path}/cc_{dataset}_java_{aug}.csv'  
        jsonl_to_csv(jsonl_file, csv_file, aug)
        aug_df = pd.read_csv(f'{aug_path}/{dataset}_java_{aug}.csv')     
        train_df = train_df.rename(columns={'index': 'idx'})
    else:    
        jsonl_file = f'{aug_path}/cc_{dataset}_{aug}.jsonl'
        csv_file = f'{aug_path}/cc_{dataset}_{aug}.csv'
        jsonl_to_csv(jsonl_file, csv_file, aug)
        aug_df = pd.read_csv(f'{aug_path}/cc_{dataset}_{aug}.csv')
    aug_df = aug_df.rename(columns={f'{aug}': 'content', 'index': 'idx'})
    aug_df['content'] = aug_df['content'].astype(str)
    aug_error_df = aug_df[aug_df['content'].str.contains('ERROR')]
    error_idxs = aug_error_df['idx'].tolist()
    aug_df = aug_df[~aug_df['content'].str.contains('ERROR')]
    
    labels = list(train_df["label"])
    num_class = len(set(labels))
    train_labeled_idxs, train_unlabeled_idxs = train_split_v1(aug_error_df, train_df, n_labeled_per_class)
    
    train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
    print("initial labeled dataset number:", len(train_l_df))
    print("initial labeled data index number:",train_l_df['idx'].value_counts().to_dict())

    aug_df = aug_df[aug_df['idx'].isin(train_l_df['idx'])]
    concat_df = pd.concat([train_l_df, aug_df], ignore_index=True)

    for idx in concat_df['idx'].unique():
        mask = (concat_df['idx'] == idx)
        concat_df.loc[mask, 'label'] = train_l_df.loc[train_l_df['idx'] == idx, 'label'].values[0]

    print("initial labeled dataset+aug number:", len(concat_df))
    print("initial labeled data+aug index number:",concat_df['idx'].value_counts().to_dict())
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


#artificial
def get_dataloader_v2(data_path, dataset, n_labeled_per_class, bs, load_mode='semi_SSL',  token = None):
    aug='forwhile'
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
    if dataset == 'corcod':
        jsonl_file = f'{aug_path}/cc_{dataset}_java_{aug}.jsonl'
        csv_file = f'{aug_path}/cc_{dataset}_java_{aug}.csv'  
        jsonl_to_csv(jsonl_file, csv_file, aug)
        aug_df = pd.read_csv(f'{aug_path}/cc_{dataset}_java_{aug}.csv')     
        train_df = train_df.rename(columns={'index': 'idx'})
    else:    
        jsonl_file = f'{aug_path}/cc_{dataset}_{aug}.jsonl'
        csv_file = f'{aug_path}/cc_{dataset}_{aug}.csv'
        jsonl_to_csv(jsonl_file, csv_file, aug)
        aug_df = pd.read_csv(f'{aug_path}/cc_{dataset}_{aug}.csv')
    aug_df = aug_df.rename(columns={f'{aug}': 'content', 'index': 'idx'})
    aug_df['content'] = aug_df['content'].astype(str)
    aug_df = aug_df[~aug_df['content'].str.contains('ERROR')]
    
    labels = list(train_df["label"])
    num_class = len(set(labels))
    
    
    train_labeled_idxs, train_unlabeled_idxs = train_split_v2(aug_df, train_df, n_labeled_per_class)
    
    train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
    
    print("initial labeled dataset number:", len(train_l_df))
    print("initial labeled data index number:",train_l_df['idx'].value_counts().to_dict())
    
    aug_df = aug_df[aug_df['idx'].isin(train_l_df['idx']+1)]
    concat_df = pd.concat([train_l_df, aug_df], ignore_index=True)
    
    for idx in concat_df['idx'].unique():
        if idx in (train_l_df['idx']+1).values:
            mask = (concat_df['idx'] == idx)
            concat_df.loc[mask, 'label'] = train_l_df.loc[train_l_df['idx']+1 == idx, 'label'].values[0]

    print("initial labeled dataset+aug number:", len(concat_df))
    print("initial labeled data+aug index number:",concat_df['idx'].value_counts().to_dict())
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

def process_augtype(augtype, aug_path, dataset):
    if dataset == 'corcod':
        jsonl_file = f'{aug_path}/cc_{dataset}_java_{augtype}.jsonl'
        csv_file = f'{aug_path}/cc_{dataset}_java_{augtype}.csv'
        jsonl_to_csv(jsonl_file, csv_file, augtype)
    else:
        jsonl_file = f'{aug_path}/cc_{dataset}_{augtype}.jsonl'
        csv_file = f'{aug_path}/cc_{dataset}_{augtype}.csv'
        jsonl_to_csv(jsonl_file, csv_file, augtype)
    
    df = pd.read_csv(csv_file)
    df = df.rename(columns={augtype: 'content', 'index': 'idx'})
    df['content'] = df['content'].astype(str)
    
    error_df = df[df['content'].str.contains('ERROR')]
    df = df[~df['content'].str.contains('ERROR')]
    
    return df, error_df

#artificial (loop translation + backtranlation )
def get_dataloader_v3(data_path, dataset, n_labeled_per_class, bs, load_mode='semi_SSL', token = None):

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
    train_df = train_df.rename(columns={'index': 'idx'})
    dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
    if dataset != 'corcod':
        train_df['idx'] = train_df['idx']+1
    aug_path = f'../data/aug/{dataset}'
    
    for augtype in ('forwhile', 'back-translation'):
        df, error_df = process_augtype(augtype, aug_path, dataset)
        
        if augtype == 'forwhile':
            forwhile_df = df
            forwhile_error_df = error_df
        elif augtype == 'back-translation':
            backtrans_df = df
            backtrans_error_df = error_df
                       
    labels = list(train_df["label"])
    num_class = len(set(labels))
    train_labeled_idxs, train_unlabeled_idxs = train_split_v3(forwhile_error_df,backtrans_error_df, train_df, n_labeled_per_class)
    
    train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
    print("initial labeled dataset number:", len(train_l_df))
    print("initial labeled data index number:",train_l_df['idx'].value_counts().to_dict())

    forwhile_df = forwhile_df[forwhile_df['idx'].isin(train_l_df['idx'])]
    backtrans_df = backtrans_df[backtrans_df['idx'].isin(train_l_df['idx'])]
    concat_df = pd.concat([train_l_df, forwhile_df], ignore_index=True)
    concat_df = pd.concat([concat_df, backtrans_df], ignore_index=True)


    for idx in concat_df['idx'].unique():
        mask = (concat_df['idx'] == idx)
        concat_df.loc[mask, 'label'] = train_l_df.loc[train_l_df['idx'] == idx, 'label'].values[0]

    print("initial labeled dataset+aug number:", len(concat_df))
    print("initial labeled data+aug index number:",concat_df['idx'].value_counts().to_dict())
    concat_df['label'] = concat_df['label'].astype(int)
    train_l_df = concat_df 
    
    # check statistics info
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


#artificial (loop translation + backtranlation )
def get_dataloader_v4(data_path, dataset, n_labeled_per_class, bs, load_mode='semi_SSL', token = None):

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
    train_df = train_df.rename(columns={'index': 'idx'})
    dev_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
    test_df = pd.read_csv(os.path.join(data_path,'test.csv'))
    if dataset != 'corcod':
        train_df['idx'] = train_df['idx']+1

    aug_path = f'../data/aug/{dataset}'
    
    for augtype in ('forwhile', 'back-translation'):
        df, error_df = process_augtype(augtype, aug_path, dataset)
        
        if augtype == 'forwhile':
            forwhile_df = df
            forwhile_error_df = error_df
        elif augtype == 'back-translation':
            backtrans_df = df
            backtrans_error_df = error_df
            
    labels = list(train_df["label"])
    num_class = len(set(labels))
    train_labeled_idxs, train_unlabeled_idxs = train_split_v4(forwhile_df,backtrans_df, train_df, n_labeled_per_class)
    
    train_l_df, train_u_df = train_df.iloc[train_labeled_idxs].reset_index(drop=True), train_df.iloc[train_unlabeled_idxs].reset_index(drop=True)
    print("initial labeled dataset number:", len(train_l_df))
    print("initial labeled data index number:",train_l_df['idx'].value_counts().to_dict())


    forwhile_df = forwhile_df[forwhile_df['idx'].isin(train_l_df['idx'])]
    backtrans_df = backtrans_df[backtrans_df['idx'].isin(train_l_df['idx'])]
    concat_df = pd.concat([train_l_df, forwhile_df], ignore_index=True)
    concat_df = pd.concat([concat_df, backtrans_df], ignore_index=True)

    for idx in concat_df['idx'].unique():
        mask = (concat_df['idx'] == idx)
        concat_df.loc[mask, 'label'] = train_l_df.loc[train_l_df['idx'] == idx, 'label'].values[0]

    print("initial labeled dataset+aug number:", len(concat_df))
    print("initial labeled data+aug index number:",concat_df['idx'].value_counts().to_dict())
    concat_df['label'] = concat_df['label'].astype(int)
    train_l_df = concat_df 
    
    # check statistics info
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
