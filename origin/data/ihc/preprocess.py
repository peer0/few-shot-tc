# read data 
import pandas as pd
import os

dataset = 'ihc'
original_file_path = f'../{dataset}/train.tsv'
datatype = 'train'

def preprocess_ihc(dataset, datatype):
    original_file_path = f'../{dataset}/{datatype}.tsv'
    df = pd.read_csv(original_file_path, sep='\t')
    df = df.rename(columns={'class': 'label', 'post': 'content'})
    
    df['label'] = df['label'].map({'not_hate': 0, 'implicit_hate': 1})
    df.to_csv(f'../{dataset}/{datatype}.tsv',sep='\t', index=False)
    return df

dataset = 'ihc'
df = preprocess_ihc(dataset, 'train')
valid_df = preprocess_ihc(dataset, 'valid')
test_df = preprocess_ihc(dataset, 'test')


print("==========================finished prepocessing==========================")
#for test
# df = df[:10]

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
df.to_csv(original_file_path,sep='\t', index=False)
print("==========================finished Strong Augmentation==========================")
print(df)
