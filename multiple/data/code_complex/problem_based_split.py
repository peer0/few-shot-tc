import pandas as pd
import numpy as np
import os

np.random.seed(44)

# jsonl 파일에서 데이터 로드
# dataset = 'java_extended_data'
dataset = 'python_extended_data'
data = pd.read_json(f'{dataset}.jsonl',lines=True)

# 데이터를 섞음
data = data.sample(frac=1, random_state=42)
# 'Problem' 컬럼을 기준으로 데이터를 그룹화
grouped = data.groupby('problem')

train_data = pd.DataFrame()
valid_data = pd.DataFrame()
test_data = pd.DataFrame()

# 각 그룹을 훈련, 검증, 테스트 세트로 분할
for name, group in grouped:

    features = group.drop('complexity', axis=1)
    labels = group['complexity']

    if len(group) == 1:
        # 샘플이 하나만 있는 경우, 랜덤하게 훈련, 검증, 테스트 중 하나에 할당
        choice = np.random.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1])
        if choice == 'train':
            train_data = pd.concat([train_data, group], ignore_index=True)
        elif choice == 'valid':
            valid_data = pd.concat([valid_data, group], ignore_index=True)
        else:
            test_data = pd.concat([test_data, group], ignore_index=True)
    else:
        # 샘플이 여러 개 있는 경우, 해당 그룹의 모든 샘플을 랜덤하게 train, valid, 또는 test 중 하나에 할당
        choice = np.random.choice(['train', 'valid', 'test'], p=[0.8, 0.1, 0.1])
        if choice == 'train':
            train_data = pd.concat([train_data, group], ignore_index=True)
        elif choice == 'valid':
            valid_data = pd.concat([valid_data, group], ignore_index=True)
        else:
            test_data = pd.concat([test_data, group], ignore_index=True)

#경로가 없다면 생성
if not os.path.exists(f'./problem_based_split/{dataset}/'):
    os.makedirs(f'./problem_based_split/{dataset}/')
    
# 데이터프레임을 CSV 파일로 저장
train_data.to_csv(f'./problem_based_split/{dataset}/train.csv', index=False)
valid_data.to_csv(f'./problem_based_split/{dataset}/dev.csv', index=False)
test_data.to_csv(f'./problem_based_split/{dataset}/test.csv', index=False)
