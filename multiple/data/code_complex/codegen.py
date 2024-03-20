import pandas as pd

# datatype 사용
dataset = 'code_complex/problem_based_split'
datatype = 'train'
# data_name = args.dataname
data_name = 'java_extended_data'
# data_name = 'python_extended_data'
original_file_path = f'../{dataset}/{data_name}/{datatype}.csv'

# (3)strong augmentation: code translation
from transformers import AutoModelForCausalLM, AutoTokenizer
df = pd.read_csv(original_file_path, sep=',')
df['back_translation'] = 0

checkpoint = "Salesforce/codegen-16B-nl"
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


def complete_code(text):
    try:
         # 모델의 명세에 기반하여 max_length 설정
        max_length = 128

        # 입력 텍스트를 모델의 max_length에 맞게 자르거나 패딩
        text = text[:max_length]

        # 토큰화 및 생성
        tokens = tokenizer(text, return_tensors="pt")
        print("토큰화된 입력 크기:", tokens['input_ids'].size())

        # 코드 생성
        completion = model.generate(**tokens, max_length=max_length, num_return_sequences=1)
        return tokenizer.decode(completion[0])
    except Exception as e:
        print(f"Error occurred: {e}")
        return f"Error: {e}"  # 예외 메시지 반환

for i, row in df.iterrows():
    df.at[i, 'back_translation'] = complete_code(row['content'])
    df.to_csv(original_file_path, mode='a', header=False, index=False)

print("==========================finished Strong Augmentation==========================")
print(df)