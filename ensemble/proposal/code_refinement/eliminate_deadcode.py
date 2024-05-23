import os
import sys
import javalang
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from AST2Code import AST2Code_module
def augmentation(path):
    file=open(f'{path}.txt').read().split('\n')[:-1]
    datas=[x.split('\t') for x in file]

    codes=[]
    complexity=[]
    for item in datas:
        complexity.append(item[0])
        codes.append(item[1])
    module2 = AST2Code_module()
    new_datas=[]
    fail=0
    for i,code in enumerate(codes):
        module = AST2Code_module(dead_code=1)
        try:
            tree=javalang.parse.parse(code)
            augmented_code=module.AST2Code(tree).replace('  ', ' ').strip().replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
            tree=javalang.parse.parse(augmented_code)
            module2.split_method(tree)
            new_datas.append(complexity[i]+'\t'+augmented_code)
        except:
            fail+=1
            new_datas.append(complexity[i]+'\t'+code)
    print(f'data:{path} fail:{fail}')
    with open(f'{path}_d.txt', 'w', encoding='utf8') as f:
        f.writelines('\n'.join(new_datas))

if __name__=='__main__':
    length_items=['256','512','1024','over']
    complexity_items=['constant','linear','quadratic','cubic','logn','nlogn','np']

    for f in range(5):
        path=f'data/test_{f}_fold'
        augmentation(path)
        
        for i in length_items:
            path='data/length_split/'+f'{i}_test_{f}_fold'
            augmentation(path)
        for i in complexity_items:
            path='data/complexity_split/'+f'{i}_test_{f}_fold'
            augmentation(path)