from main_SSL_load import multiRun
import sys
import os

########################################중요 model이름 매번 바꿔서 넣어야함.



# token, net_arch만 변경해서 이용하면됨.

### code_complex

n_labeled_per_class = int(sys.argv[1]) #few shot 수
bs = 7  # 4, 8 # batch size
# ul_ratio = {number of unlabels to check}
psl_threshold_h = float(sys.argv[4])
lr = float(sys.argv[3])

weight_u_loss = 1
load_mode = 'semi_SSL'

labeling_mode = 'hard'
adaptive_threshold = True

num_nets = 1
cross_labeling  = False

weight_disagreement = True
disagree_weight = 0.9 

ema_mode = False
ema_momentum = 0.9

seeds_list = [43]
device_idx = 0
val_interval = 25
early_stop_tolerance = int(sys.argv[5])
max_epoch = 100
max_step = 100000   

model_name = sys.argv[2]
if model_name=='codebert':
        net_arch = "microsoft/codebert-base"
elif model_name=='unixcoder':
        net_arch = "microsoft/unixcoder-base"
elif model_name=='codet5p':
        net_arch = "Salesforce/codet5p-110m-embedding"





experiment_home = './experiment/'  #저장소 path
dataset = sys.argv[6]
# JointMatch
num_runs = 1
num_nets = 1


print(dataset.split('/')[3])
save_name = f"{n_labeled_per_class}_{net_arch.split('/')[1]}_{lr}_{psl_threshold_h}_{early_stop_tolerance}_{dataset.split('/')[3]}"


print("save_name: {}".format(save_name))

adaptive_threshold = True
weight_disagreement = 'True'

multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
        n_labeled_per_class=n_labeled_per_class, bs=bs, lr=lr,seeds_list=seeds_list,
        load_mode=load_mode,
        weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
        num_nets=num_nets, cross_labeling=cross_labeling, 
        weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
        ema_mode=ema_mode, ema_momentum=ema_momentum,
        val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step, 
        max_epoch = max_epoch, save_name = save_name, net_arch = net_arch, labeling_mode = labeling_mode) 
