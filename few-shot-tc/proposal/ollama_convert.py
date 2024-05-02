import ollama
import os
import json
import argparse
from tqdm import tqdm

def ollama_chat(code, model):
    prompt = '''
    With the following code snippet, please generate its variant code snippet while maintaining its functionality.
    Instead of summarizing the code snippet's functionality, provide a modified version of this code snippet.
    The programming language of the variant code snippet must be the same as the programming language of the input code snippet.
    Please provide only the code snippet.
    Do not provide the explanation of the modified code snippet.
    '''

    instruction_message = prompt + code

    result = {}
    try:
        ### Chat 
        # message = {'role': 'user', 'content': instruction_message}
        # response =  ollama.chat(model=model, messages=[message])
        # result["src_llm"] = response['message']['content']

        ### Generate
        response = ollama.generate(model=model, prompt=instruction_message)
        result["src_llm"] = response['response']

    except Exception as e:
        print("Error in inference.")
        print(f"Error was : {e}")
        result["error_message"] = str(e)
        result["src_llm"] = "ERROR"
    return result

def main(args):
    file_name = args.inputfile + '/' + args.filename + '_code.jsonl' 
    code_json = open(file_name).readlines()
    code_contents = []
    for i in code_json:
        code_contents.append(json.loads(i))    

    save_model_path = "{}".format(args.model)
    all_data = []

    checkpoint_len=0
    checkpoint_index=0
    checkpoint_data=''
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    else:
        if os.path.isfile("{}/{}.predictions.jsonl".format(save_model_path,args.filename)):
            with open("{}/{}.predictions.jsonl".format(save_model_path,args.filename),'r') as f:
                all_data = f.readlines()
                last_data = json.loads(all_data[-1])
                checkpoint_len = len(all_data)
                checkpoint_index = last_data["idx"]
                checkpoint_data = last_data["src"]

            assert code_contents[checkpoint_len-1]["src"]==checkpoint_data, "code_contents: {}\ncheckpoint_data: {}".format(code_contents[checkpoint_len-1]["src"],checkpoint_data)
            assert code_contents[checkpoint_len-1]["idx"]==checkpoint_index, "code_contents: {}\ncheckpoint_data: {}".format(code_contents[checkpoint_len-1]["idx"],checkpoint_data)
            code_contents = code_contents[checkpoint_len:]

    with open("{}/{}.predictions.jsonl".format(save_model_path,args.filename),'w') as f:
        for data_instance in all_data:
            json.dump(json.loads(data_instance),f)
            f.write('\n')

        pbar = tqdm(total=len(code_contents),desc="Processing")
        for code_content in code_contents:
            result = ollama_chat(code_content["src"], args.model)
            try: 
                result["src_human"] = code_content["src"]
                result["idx"] = code_content["idx"]
            except Exception as e:
                print(f"ERROR: {e} with result {result} for content: {code_content}.")
                result["error_message"] = str(e)
                result = {"src_llm": "ERROR"}
            result["llm"] = args.model
            json.dump(result,f)
            f.write('\n')
            pbar.update(1)            
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', '-m',
        type=str,
        help="The LLM used for the inference",
        required=True,
        dest='model'
    )
    parser.add_argument(
        '--inputfile', '-i',
        type=str,
        help="The input file path",
        required=True,
        dest='inputfile'
    )
    parser.add_argument(
        '--filename', '-fn',
        type=str,
        help="C or C++ or Java or Python",
        required=True,
        dest='filename'
    )

    args = parser.parse_args()

    main(args)