import pandas as pd
from sklearn.model_selection import train_test_split
import os

# CSV 파일에서 데이터 로드
dataset = 'CorCoD'
# dataset = 'java_extended_data'
data = pd.read_json(f'./corcod.index.jsonl',lines=True)

# 데이터를 feature과 레이블로 분할
features = data.drop('complexity', axis=1)
labels = data['complexity']

# 재현 가능성을 위해 랜덤 시드 설정
random_state = 42

# 데이터를 훈련 및 임시(검증 + 테스트) 세트로 분할
# 전체 데이터의 20퍼센트를 테스트 세트로 지정
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.2, random_state=random_state, stratify=labels
)

# 임시 세트를 검증 및 테스트 세트로 분할
#임시(검증+테스트)데이터의 50%를 검증 세트로, 나머지 50%를 테스트 세트로 지정
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
)

# 훈련, 검증 및 테스트 세트를 위한 데이터프레임 생성
train_data = pd.concat([X_train, y_train], axis=1)
valid_data = pd.concat([X_valid, y_valid], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

#경로가 없다면 생성
if not os.path.exists(dataset):
    os.makedirs(dataset)
    
# 데이터프레임을 CSV 파일로 저장
train_data.to_csv(f'./train.csv', index=False)
valid_data.to_csv(f'./dev.csv', index=False)
test_data.to_csv(f'./test.csv', index=False)
