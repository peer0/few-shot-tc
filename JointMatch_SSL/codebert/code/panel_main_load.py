from main_SSL_load import multiRun
########################################중요 model이름 매번 바꿔서 넣어야함.



# token, net_arch만 변경해서 이용하면됨.

### code_complex
n_labeled_per_class = 10 #few shot 수
bs = 7  # 4, 8 # batch size
#ul_ratio = 554            # 10shot 이면 549 , 5shot이면 554, 1shot이면 558 # 현재 전체 데이터로 자동으로 설정됨


#token = "microsoft/codebert-base"
#token = "microsoft/unixcoder-base"
token = "Salesforce/codet5p-110m-embedding"

#net_arch = 'microsoft/codebert-base'
#net_arch = "microsoft/unixcoder-base"
net_arch = "Salesforce/codet5p-110m-embedding"

labeling_mode = 'soft'

lr = 1e-5  # 
weight_u_loss = 1
psl_threshold_h = 0.7 # ul의 predict의 임계값
adaptive_threshold = True

max_epoch = 100

#num_nets = 2 # joint match이기에 2개의 model을 이용해서 2. 
num_nets = 1 # 여기에서는 SSL을 위한 실험이기에 1개의 모델만 이용함.

cross_labeling  = False # 여기에서는 SSL을 위한 실험이기에 FALSE 이용함.

weight_disagreement = True
disagree_weight = 0.9 

ema_mode = False

ema_momentum = 0.9
val_interval = 25  # 몇번째 만큼 검증을 하고 모델이 어떤지 파악하는 parameter
early_stop_tolerance = 10
p_tolerance = 10
max_step = 100000   

device_idx = 0
experiment_home = './experiment/few_shot'  #저장소 path
dataset = '../codecomplex_java'   # 'ag_news', 'yahoo', 'imdb', 'empatheticdialogues', 'threecrises', 'goemotions'
# seeds_list = [0]
# JointMatch
num_runs = 1 # 같은 실험
#num_nets = 2 # model 수.
num_nets = 1 # model 수. # 여기에서는 SSL을 위한 실험이기에 1개의 모델만 이용함.

save_name = "{}_{}_{}_{}".format(dataset.split('/')[1],net_arch.split('/')[1],n_labeled_per_class,psl_threshold_h)

#cross_labeling = True
adaptive_threshold = True
weight_disagreement = 'True'

multiRun(device_idx=device_idx, experiment_home=experiment_home, dataset=dataset, num_runs=num_runs,
        n_labeled_per_class=n_labeled_per_class, bs=bs, lr=lr,
        weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
        num_nets=num_nets, cross_labeling=cross_labeling, 
        weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
        ema_mode=ema_mode, ema_momentum=ema_momentum,
        val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=max_step, 
        max_epoch = max_epoch, save_name = save_name, token = token, net_arch = net_arch, labeling_mode = labeling_mode) # ul_ratio 자동 설정으로 뺌
