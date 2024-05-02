import pandas as pd

# datatype 사용
# dataset = 'code_complex/problem_based_split'
dataset = 'code_complex/random_split'
datatype = 'train'
# data_name = args.dataname
data_name = 'java_extended_data'
# data_name = 'python_extended_data'
original_file_path = f'../{dataset}/{data_name}/{datatype}.csv'


def preprocess_ihc(dataset, datatype):
    original_file_path = f'../{dataset}/{data_name}/{datatype}.csv'
    df = pd.read_csv(original_file_path)
    df = df.rename(columns={'complexity': 'label', 'src': 'content'})
    
    df['label'] = df['label'].map({ 'constant':1, 'logn':2, 'linear':3, 'nlogn':4, 'quadratic':5, 'cubic':6, 'np':7})
    df.to_csv(f'../{dataset}/{data_name}/{datatype}.csv', index=False)
    return df
 
# preprocess_ihc(dataset, 'train') #java_extended_data
# preprocess_ihc(dataset, 'dev')
# preprocess_ihc(dataset, 'test')


# print("==========================finished prepocessing==========================")

# # (1) Weak Augmentation: Synonym Replacement
# import nlpaug.augmenter.word as naw
# from tqdm import tqdm

df = pd.read_csv(original_file_path, sep=',')
# df['synonym_aug'] = 0
# aug = naw.SynonymAug(aug_src='wordnet')
# for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#     df['synonym_aug'][idx] = aug.augment(row['content'])[0]
    
# df.to_csv(original_file_path,index=False)
# print("==========================finished Weak Augmentation==========================")
# print(df)


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
df.to_csv(original_file_path,sep='\t',index=False)
print("==========================finished Strong Augmentation==========================")
print(df)

# (3)strong augmentation: code translation
# from transformers import AutoModelForCausalLM, AutoTokenizer
# df = pd.read_csv(original_file_path, sep=',')
# df['back_translation'] = 0
# def complete_code(text):
#     checkpoint = "Salesforce/codegen-350M-mono"
#     model = AutoModelForCausalLM.from_pretrained(checkpoint)
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     text = text[:2048]
#     completion = model.generate(**tokenizer(text, return_tensors="pt"), pad_token_id=tokenizer.eos_token_id, max_new_tokens=1000)

#     return tokenizer.decode(completion[0])


# df['back_translation'] = df['content'].apply(complete_code)

# # 결과를 CSV 파일로 저장
# df.to_csv(original_file_path, index=False)
# print("==========================finished Strong Augmentation==========================")
# print(df)
