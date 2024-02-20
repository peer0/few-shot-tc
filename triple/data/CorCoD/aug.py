# read data 
import pandas as pd
import os
import argparse

# ArgumentParser 객체 생성
# parser = argparse.ArgumentParser(description='Process some integers.')

# # datatype 인자 추가
# parser.add_argument('--dataname', type=str, help='an integer for the accumulator')

# # 인자 파싱
# args = parser.parse_args()

# datatype 사용
dataset = 'CorCoD'
datatype = 'train'
# data_name = args.dataname
# data_name = 'java_extended_data'
# data_name = 'python_extended_data'
original_file_path = f'../{dataset}/{datatype}_t.csv'


def preprocess_ihc(dataset, datatype):
    original_file_path = f'../{dataset}/{datatype}_t.csv'
    df = pd.read_csv(original_file_path)
    df = df.rename(columns={'complexity': 'label', 'src': 'content'})

    df['label'] = df['label'].map({'1':1,'logn':2,'n':3,'n_square':4,'nlogn':5})
    df.to_csv(f'../{dataset}/{datatype}.csv', index=False)
    return df

df = preprocess_ihc(dataset, 'train') #java_extended_data
valid_df = preprocess_ihc(dataset, 'dev')
test_df = preprocess_ihc(dataset, 'test')


print("==========================finished prepocessing==========================")

# (1) Weak Augmentation: Synonym Replacement
import nlpaug.augmenter.word as naw
from tqdm import tqdm


df['synonym_aug'] = 0
aug = naw.SynonymAug(aug_src='wordnet')
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['synonym_aug'][idx] = aug.augment(row['content'])[0]
    
# df.to_csv(original_file_path,sep='\t',index=False)
print("==========================finished Weak Augmentation==========================")
print(df)


# (2) Strong Augmentation: Back Translation
import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)
df['back_translation'] = 0
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en',
    device=device
)

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['back_translation'][idx] = back_translation_aug.augment(row['content'])[0]
df.to_csv(f'../{dataset}/{data_name}/{datatype}_aug.csv', index=False)
print("==========================finished Strong Augmentation==========================")
print(df)
