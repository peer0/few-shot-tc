import pandas as pd
import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm
import torch


original_file_path = './data.csv'
save_home = './'
datatype = 'train'
df = pd.read_csv(original_file_path)
print(df.head())


# (1) Weak Augmentation: Synonym Replacement
df['synonym_aug'] = 0
aug = naw.SynonymAug(aug_src='wordnet')
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['synonym_aug'][idx] = aug.augment(row['content'])[0]
df.to_csv(save_home + datatype + ".csv",index=False)
# print saving path
print('Data Augmentation Done! Check the saved file at: ', save_home + datatype + ".csv")


# (2) Strong Augmentation: Back Translation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)
# file = "unlabeled_data.csv"
# df = pd.read_csv(file)
df['back_translation'] = 0
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en',
    device=device
)
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['back_translation'][idx] = back_translation_aug.augment(row['content'])[0]
df.to_csv(save_home + datatype + ".csv",index=False)
# print saving path
print('Data Augmentation Done! Check the saved file at: ', save_home + datatype + ".csv")