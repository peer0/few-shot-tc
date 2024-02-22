import pandas as pd
import pickle
import argparse
import numpy as np
import random
import os
import code_tokenize as ctok

np.random.seed(0)
random.seed(0)



def preprocess_data(dataset):
    os.makedirs('preprocessed_data', exist_ok=True)

    if dataset == 'corcod':
        class2int = {'linear':1, 'quadratic':2, 'nlogn':3, 'constant':4, 'logn':5}
        data_dict = {}
        data_home = 'dataset/corcod/'

        for datatype in ['train', 'dev', 'test']:
            datafile = data_home + datatype + '.csv'
            data = pd.read_csv(datafile)

            label, code = [], []

            for i,one_class in enumerate(data['complexity']):
                label.append(class2int[one_class])
                code.append(data['src'][i])

            print('Tokenizing data...')
            tokenized_code = [ctok.tokenize(k, lang="python") for k in code]
            processed_data = {}
            processed_data['tokenized_code'] = tokenized_code
            processed_data['label'] = label
            processed_data['code'] = code

            processed_data = pd.DataFrame.from_dict(processed_data)
            data_dict[datatype] = processed_data

        with open('./preprocessed_data/corcod_preprocessed_ctok.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()


    elif dataset == 'cc_py':        # codecomplex_python
        class2int = {'constant':1, 'logn':2, 'linear':3, 'nlogn':4, 'quadratic':5, 'cubic':6, 'np':7}
        data_dict = {}
        data_home = 'dataset/codcomplex_python/'

        for datatype in ['train', 'dev', 'test']:
            datafile = data_home + datatype + '.csv'
            data = pd.read_csv(datafile)

            label, code = [], []

            for i,one_class in enumerate(data['complexity']):
                label.append(class2int[one_class])
                code.append(data['src'][i])

            print('Tokenizing data...')
            tokenized_code = [ctok.tokenize(k, lang="python") for k in code]
            processed_data = {}
            processed_data['tokenized_code'] = tokenized_code
            processed_data['label'] = label
            processed_data['code'] = code

            processed_data = pd.DataFrame.from_dict(processed_data)
            data_dict[datatype] = processed_data

        with open('./preprocessed_data/cc_py_preprocessed_ctok.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()


    elif dataset == 'cc_java':        # codecomplex_java
        class2int = {'constant':1, 'logn':2, 'linear':3, 'nlogn':4, 'quadratic':5, 'cubic':6, 'np':7}
        data_dict = {}
        data_home = 'dataset/codcomplex_java/'

        for datatype in ['train', 'dev', 'test']:
            datafile = data_home + datatype + '.csv'
            data = pd.read_csv(datafile)

            label, code = [], []

            for i,one_class in enumerate(data['complexity']):
                label.append(class2int[one_class])
                code.append(data['src'][i])

            print('Tokenizing data...')
            tokenized_code = [ctok.tokenize(k, lang="java", syntax_error = "ignore") for k in code]
            processed_data = {}
            processed_data['tokenized_code'] = tokenized_code
            processed_data['label'] = label
            processed_data['code'] = code

            processed_data = pd.DataFrame.from_dict(processed_data)
            data_dict[datatype] = processed_data

        with open('./preprocessed_data/cc_java_preprocessed_ctok.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
            f.close()

    else:
        NotImplementedError




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default='corcod', type=str)
    args = parser.parse_args()

    preprocess_data(args.d)
    