import pandas as pd

# datatype 사용
dataset = 'code_contests'
datatype = 'train'


original_file_path = f'../{dataset}/{datatype}.csv'

# (3)strong augmentation: code translation
from transformers import AutoModelForCausalLM, AutoTokenizer
df = pd.read_csv(original_file_path, sep=',')
df['back_translation'] = 0


def complete_code(text):
    checkpoint = "Salesforce/codegen-16B-nl"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

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



# 수정된 함수 호출
df['back_translation'] = df['content'].apply(complete_code)

# 결과를 CSV 파일로 저장
df.to_csv(original_file_path, index=False)
print("==========================finished Strong Augmentation==========================")
print(df)