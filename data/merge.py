import pandas as pd
import json

# Load the CSV file into a DataFrame
lang = 'python'
csv_file_path = f'{lang}/train.csv'
train_df = pd.read_csv(csv_file_path)

# Read the JSONL file and map src to back-translation
jsonl_back_path = f'aug/{lang}/cc_python_back-translation.jsonl'
jsonl_loop_path = f'aug/{lang}/cc_python_forwhile.jsonl'
src_to_back_translation = {}
src_to_loop_translation = {}

with open(jsonl_back_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        json_data = json.loads(line)
        src_to_back_translation[json_data['src']] = json_data['back-translation']

with open(jsonl_loop_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        json_data = json.loads(line)
        src_to_loop_translation[json_data['src']] = json_data['forwhile']

# Add a new column to the DataFrame for back-translation
train_df['back-translation'] = train_df['content'].map(src_to_back_translation)
train_df['forwhile'] = train_df['content'].map(src_to_loop_translation)

# Save the updated DataFrame to a new CSV file
train_df.to_csv(f'{lang}/cross_train.csv', index=False)

print(f"Updated CSV file has been saved to {lang}/cross_train.csv")
