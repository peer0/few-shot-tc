from main import multiRun
# ## code_complex
n_labeled_per_class = 10 #few shot 수
bs = 7  # 4, 8 # batch size
ul_ratio = 10 #
lr = 1e-5  # 
weight_u_loss = 1
psl_threshold_h = 0.98 
adaptive_threshold = True
num_nets = 2 # joint match이기에 2개의 model을 이용해서 2.
cross_labeling  = True
weight_disagreement = True
disagree_weight = 0.9 
ema_mode = False 
ema_momentum = 0.9
val_interval = 25 
early_stop_tolerance = 10
max_step = 100000   

device_idx = 0
experiment_home = './experiment/test'  #저장소 path
dataset = '../data_split'   # 'ag_news', 'yahoo', 'imdb', 'empatheticdialogues', 'threecrises', 'goemotions'
# seeds_list = [0]

# JointMatch
num_runs = 1 # 같은 실험
num_nets = 2 # model 수.
cross_labeling = True
adaptive_threshold = True
weight_disagreement = 'True'

multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
        n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
        weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
        num_nets=num_nets, cross_labeling=cross_labeling, 
        weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
        ema_mode=ema_mode, ema_momentum=ema_momentum,
        val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step)


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