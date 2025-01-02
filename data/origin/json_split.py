import jsonlines
from sklearn.model_selection import train_test_split

# JSON 파일 경로
file_path = '/home/imsuhan22/few-shot-tc/JointMatch_SSL/codebert/data/corcod.index.jsonl'
# file_path = '/home/sunny5574/hsan/dataset/raw/java_extended_data.jsonl'
# file_path = '/home/sunny5574/hsan/dataset/raw/python_extended_data.jsonl'

# 데이터 불러오기
data = []
with jsonlines.open(file_path, 'r') as reader:
    for line in reader:
        data.append(line)

# 데이터를 train, dev, test로 분할
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 결과 출력 for dataset statistics
print(f"Train set size: {len(train_data)}")
print(f"Dev set size: {len(dev_data)}")
print(f"Test set size: {len(test_data)}")

# 분할된 데이터를 각각의 파일로 저장
data_name = 'corcod'
# data_name = 'cc_java'
# data_name = 'cc_py'

with jsonlines.open(f'{data_name}_train.jsonl', 'w') as writer:
    writer.write_all(train_data)

with jsonlines.open(f'{data_name}_dev.jsonl', 'w') as writer:
    writer.write_all(dev_data)

with jsonlines.open(f'{data_name}_test.jsonl', 'w') as writer:
    writer.write_all(test_data)