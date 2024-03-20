import os
import pandas as pd

# CSV 파일이 있는 경로
path = "/home/jungin/workspace/JointMatch/multiple/data/code_contests"

# 경로에서 CSV 파일 목록 가져오기
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')and f.startswith('train-')]

# 각 CSV 파일을 로드하고, language가 4인 행만 필터링
dataframes = []
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(path, csv_file))
    df = df[df['language'] == 4]
    dataframes.append(df)

# 모든 DataFrame을 하나로 합치기
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df.rename(columns={'solutions': 'content'})

# 결과를 CSV 파일로 저장
combined_df.to_csv('train.csv', index=False)
print(combined_df)

print("==========================finished prepocessing==========================")