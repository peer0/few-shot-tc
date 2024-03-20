import re
import time
import os
import sys
import json
import argparse
from tqdm import tqdm
from collections import OrderedDict




def main(args):
    directory = os.path.dirname(args.inputfile)
    code_json = open(args.inputfile).readlines()
    save_model_path = "{}-{}".format(args.model,args.index)
    code_contents = []
    all_data = []

    for i in code_json:
        code_contents.append(json.loads(i))


    complexity_class = ['constant', 'logn', 'linear', 'nlogn','quadratic', 'cubic', 'exponential']
    complexity_class.reverse()
    print(complexity_class)
    checkpoint_len=0
    checkpoint_index=0
    checkpoint_data=''
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    pbar = tqdm(total=len(code_contents),desc="{} inferenceing".format(args.model))
    with open("{}/{}.{}.{}.predictions.jsonl".format(save_model_path,args.filename,args.tool,args.complexity_type),'w') as f:
        for i,content in enumerate(code_contents):
            answer = content.pop("complexity") # pop the 'answer' element
            retrynum = 0
            resulting_json=''
            while(retrynum< 3):
                try:
                    resulting_json = LLMParadigmTranslator(content,args.model,args.tool)
                    resulting_json = json.loads(resulting_json)
                    resulting_json['src'] = content['src']
                    resulting_json['answer'] = answer
                    break
                except Exception as e:
                    print("#"*10+"{}".format(i)+"#"*10)
                    print(f"Error was : {e}")
                    print(f"resulting json was: {resulting_json}")
                    retrynum+=1
                    if retrynum==3:
                        resulting_json = {"src":content["src"], "answer": answer, "complexity": "ERROR", "reason":"ERROR"}
            
            resulting_json["idx"] = content["idx"]
            json.dump(resulting_json,f)
            f.write('\n')
            pbar.update(1)

    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inputfile', '-i',
        type=str,
        help="The input file path",
        required=True,
        dest='inputfile'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help="The models used as the baselines",
        required=True,
        dest='model'
    )
    parser.add_argument(
        '--filename', '-fn',
        type=str,
        help="Java or Python or Corcod",
        required=True,
        dest='filename'
    )
    parser.add_argument(
        '--type', '-t',
        type=str,
        help='''
        There are xx types of the learning mechanism:
        1) model-central: only by the model; basic self-training,
        2) rule-centric : use the pre-defined rules,
        1) combined-ensemble: use both modules simultaneously to pseudo-label,
        2) procedural-ensemble: use both modules respectively to pseudo-label.
        ''',
        required=True,
        dest='type'
    )
    parser.add_argument(
        '--trial', '-tn',
        type=int,
        help='''
        The number of trials.
        ''',
        required=True,
        dest='trial'
    )
    parser.add_argument(
        '--threshold', '-th',
        type=float,
        help='''
        The threshold confidence score.
        ''',
        required=True,
        dest='threshold'
    )
    
    args = parser.parse_args()
    main(args)
