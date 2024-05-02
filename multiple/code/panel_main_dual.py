#main_v2.py
#1.seed_list 추가 : 0만
from main_multiple import multiRun
# ## code_complex
n_labeled_per_class = 10
bs = 7  # 4, 8
ul_ratio = 10 #label_data의몇배가 unlabel_data로 사용되는지
lr = 1e-5 
weight_u_loss = 1 #unsupervised loss weight
psl_threshold_h = 0.98 #Fixed threshold(타우)
adaptive_threshold = True #adaptive_thresthold를 적용할 것인가
num_nets = 2 #모델개수
cross_labeling  = True 
weight_disagreement = True
disagree_weight = 0.9 
ema_mode = False 
ema_momentum = 0.9 #EMA decay
val_interval = 25 
early_stop_tolerance = 10
max_step = 100000    
seeds_list = [0] #2024-01-29추가
labeling_mode = 'soft' #2024-01-30추가
# 2024-02-21추가
# net_arch = 'microsoft/codebert-base'
# net_arch = 'microsoft/unixcoder-base'
# net_arch = 'Salesforce/codet5p-110m-embedding'
net_arch = ['microsoft/codebert-base','microsoft/codebert-base']

device_idx = 0
# experiment_home = './experiment/cc_rand_java_dual_syn_back'
experiment_home = './experiment/cc_rand_java_dual_trans_back'
# experiment_home = './experiment/cc_rand_java_dual_origin'
# experiment_home = './experiment/cc_pb_java_dual_origin'
dataset = 'code_complex/random_split/java_extended_data' 
# dataset = 'code_complex/problem_based_split/java_extended_data' 

# JointMatch
num_runs = 1
num_nets = 2
cross_labeling = True
adaptive_threshold = True
weight_disagreement = True

multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
        n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
        weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
        num_nets=num_nets, cross_labeling=cross_labeling, 
        weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
        ema_mode=ema_mode, ema_momentum=ema_momentum,
        val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step,
        seeds_list=seeds_list,labeling_mode=labeling_mode,net_arch=net_arch)


#for test
##########################
## AG_News
# n_labeled_per_class = 10
# bs = 8  # 4, 8
# ul_ratio = 10
# lr = 1e-5 
# weight_u_loss = 1
# psl_threshold_h = 0.98 
# adaptive_threshold = True
# num_nets = 2
# cross_labeling  = True
# weight_disagreement = True
# disagree_weight = 0.9 
# ema_mode = False 
# ema_momentum = 0.9
# val_interval = 25 
# early_stop_tolerance = 10
# max_step = 100000   

# device_idx = 0
# experiment_home = './experiment/test'
# dataset = 'ag_news'   # 'ag_news', 'yahoo', 'imdb', 'empatheticdialogues', 'threecrises', 'goemotions'

# # JointMatch
# num_runs = 5
# num_nets = 2
# cross_labeling = True
# adaptive_threshold = True
# weight_disagreement = 'True'

# multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
#         n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
#         weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
#         num_nets=num_nets, cross_labeling=cross_labeling, 
#         weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
#         ema_mode=ema_mode, ema_momentum=ema_momentum,
#         val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step)



##########################


###########################
# ## AG_News
# n_labeled_per_class = 10
# bs = 8  # 4, 8
# ul_ratio = 10
# lr = 1e-5 
# weight_u_loss = 1
# psl_threshold_h = 0.98 
# adaptive_threshold = True
# num_nets = 2
# cross_labeling  = True
# weight_disagreement = True
# disagree_weight = 0.9 
# ema_mode = False 
# ema_momentum = 0.9
# val_interval = 25 
# early_stop_tolerance = 10
# max_step = 100000   

# device_idx = 0
# experiment_home = './experiment/ag_news'
# dataset = 'ag_news'   # 'ag_news', 'yahoo', 'imdb', 'empatheticdialogues', 'threecrises', 'goemotions'

# # JointMatch
# num_runs = 5
# num_nets = 2
# cross_labeling = True
# adaptive_threshold = True
# weight_disagreement = 'True'

# multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
#         n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
#         weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
#         num_nets=num_nets, cross_labeling=cross_labeling, 
#         weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
#         ema_mode=ema_mode, ema_momentum=ema_momentum,
#         val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step)



###########################
## Yahoo
# n_labeled_per_class = 20
# bs = 4 
# ul_ratio = 10
# lr = 2e-5
# weight_u_loss = 1
# psl_threshold_h = 0.98  # 0.98, 0.99 #타우:수도라벨 threshold
# adaptive_threshold = True #동적threshold, adaptive local threshold
# num_nets = 2 #모델 수
# cross_labeling  = True
# weight_disagreement = True
# disagree_weight = 0.9
# ema_mode = False
# ema_momentum = 0.9
# val_interval = 25
# early_stop_tolerance = 10
# max_step = 100000


# device_idx = 0
# experiment_home = './experiment/yahoo'
# dataset = 'yahoo'   # 'ag_news', 'yahoo', 'imdb'

# # JointMatch
# num_runs = 5
# num_nets = 2
# cross_labeling = True
# adaptive_threshold = True
# weight_disagreement = 'True'

# multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
#         n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
#         weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
#         num_nets=num_nets, cross_labeling=cross_labeling, 
#         weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
#         ema_mode=ema_mode, ema_momentum=ema_momentum,
#         val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step)



# ###########################
# ## IMDB
# n_labeled_per_class = 10
# bs = 4 
# ul_ratio = 10
# lr = 2e-5
# weight_u_loss = 0.05
# psl_threshold_h = 0.98
# adaptive_threshold = True
# num_nets = 2
# cross_labeling  = True
# weight_disagreement = True
# disagree_weight = 0.9
# ema_mode = True
# ema_momentum = 0.9
# val_interval = 25
# early_stop_tolerance = 10
# max_step = 100000
# seeds_list = [2, 3, 4, 5, 9]


# device_idx = 0
# experiment_home = './experiment/imdb'
# dataset = 'imdb'   # 'ag_news', 'yahoo', 'imdb'

# # JointMatch
# num_runs = 5
# num_nets = 2
# cross_labeling = True
# adaptive_threshold = True
# weight_disagreement = 'True'

# multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
#         n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
#         weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
#         num_nets=num_nets, cross_labeling=cross_labeling, 
#         weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
#         ema_mode=ema_mode, ema_momentum=ema_momentum,
#         val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step, seeds_list=seeds_list)