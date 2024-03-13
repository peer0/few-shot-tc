import pandas as pd

# datatype 사용
dataset = 'code_contests'
datatype = 'train'

original_file_path = f'../{dataset}/{datatype}.csv'


# # (1) Weak Augmentation: Synonym Replacement
import nlpaug.augmenter.word as naw
from tqdm import tqdm

df = pd.read_csv(original_file_path, sep=',')
df['synonym_aug'] = 0
aug = naw.SynonymAug(aug_src='wordnet')
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['synonym_aug'][idx] = aug.augment(row['content'])[0]
    
df.to_csv(original_file_path,index=False)
print("==========================finished Weak Augmentation==========================")
print(df)
