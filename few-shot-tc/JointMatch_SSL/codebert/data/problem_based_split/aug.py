import pandas as pd

# datatype 사용
dataset = '.'

#datatype = 'train'

data_name = 'corcod.index'
#data_name = 'java_extended_data'
#data_name = 'python_extended_data'

#original_file_path = f'../{dataset}/{data_name}/{datatype}.csv'


def preprocess_ihc(dataset, datatype):
    original_file_path = f'./{dataset}/{data_name}/{datatype}.csv'
    df = pd.read_csv(original_file_path)
    df = df.rename(columns={'complexity': 'label', 'src': 'content'})
    df['label'] = df['label'].map({ 'constant':1, 'logn':2, 'linear':3, 'nlogn':4, 'quadratic':5, 'cubic':6, 'np':7})
    import pdb; pdb.set_trace() 
    df.to_csv(f'./{dataset}/{data_name}/{datatype}.csv', index=False)
    return df


preprocess_ihc(dataset, 'train') #java_extended_data
preprocess_ihc(dataset, 'dev')
preprocess_ihc(dataset, 'test')


print("==========================finished prepocessing==========================")