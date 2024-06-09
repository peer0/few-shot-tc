from main_original2 import multiRun
import sys
import os
import pdb

#n_labeled_per_class = 10
n_labeled_per_class = int(sys.argv[1])

# 추가됨
model_list = sys.argv[2].split(',')
tokens = []
net_archs = []

for model_name in model_list:
    if model_name == 'codesage':
        token = net_arch = "codesage/codesage-base"     
    elif model_name == 'codebert':
        token = net_arch = "microsoft/codebert-base"
    elif model_name == 'codet5p':
        token = net_arch = "Salesforce/codet5p-110m-embedding"
    elif model_name == 'graphcodebert':
        token = net_arch = "microsoft/graphcodebert-base"
    elif model_name == 'unixcoder':
        token = net_arch = "microsoft/unixcoder-base"
    elif model_name == 'ast-t5':
        token = net_arch = "gonglinyuan/ast_t5_base"
    elif model_name == 'codellama':
        token = net_arch = "codellama/CodeLlama-7b-hf"
    elif model_name == "starcoder":
        token = net_arch = "bigcode/starcoder"
    elif model_name == "deepseek":
        token = net_arch = "deepseek-ai/deepseek-coder-6.7b-base"

    tokens.append(token)
    net_archs.append(net_arch)
    



#lr = 1e-5 
#lr = float(sys.argv[3]) 
lr_list = sys.argv[3]
lr = [float(lr) for lr in lr_list.split()]


#psl_threshold_h = 0.98 
psl_threshold_h = float(sys.argv[4])

max_epoch = int(sys.argv[5])

#dataset = 'ag_news'   # 'ag_news', 'yahoo', 'imdb', 'empatheticdialogues', 'threecrises', 'goemotions'
dataset = sys.argv[6]

#experiment_home = './experiment/ag_news'
experiment_home = f"./experiment/{dataset.split('/')[2]}"

#추가됨
language = {dataset.split('/')[3]}
ln_list = list(language) # 이미 실험이 많이 진행되어 변경을 이렇게함.
#bs = 8  # 4, 8
#pdb.set_trace()
if ln_list[0] == 'corcode':
    bs = 7
    print('\nbs for corcod => ',bs)
else:
    bs = 7
    print('\nbs => ',bs)

if n_labeled_per_class == 5:
    val_interval = 5
else:
    val_interval = 10

####################################################
ul_ratio = 1
weight_u_loss = 1
disagree_weight = 0.9 
ema_mode = False 
ema_momentum = 0.9

early_stop_tolerance = 100000
max_step = 100000   
device_idx = 0

# JointMatch
num_runs = 1
num_nets = 2
cross_labeling = True
adaptive_threshold = True
weight_disagreement = 'True'


seeds_list = [int(sys.argv[7])]
# 추가
print('Data set ->', dataset.split('/')[2])
save_name = f"{n_labeled_per_class}_{model_list}_{lr}_{psl_threshold_h}_{dataset.split('/')[2]}_{max_epoch}_{seeds_list}_{language}"
print("save_name: {}".format(save_name))

multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
        n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
        weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
        num_nets=num_nets, cross_labeling=cross_labeling, 
        weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
        ema_mode=ema_mode, ema_momentum=ema_momentum,
        val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step,
        
        max_epoch = max_epoch, language = language, save_name = save_name, token = tokens, net_arch = net_archs,
        seeds_list = seeds_list, model_name = model_name) #추가된 파라미터