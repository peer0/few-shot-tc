import os
import random
import sys
import pdb
import numpy as np
import pandas as pd
import preprocessor as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


#### Set Path ####
# go to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = '../' #origin/
root_code = '../code' #origin/code
sys.path.append(root)
sys.path.append(root_code)
print('current work directory: ', os.getcwd())



## psudolabel data추가하는 함수. train, unlabel, mask_list,
#unlabel에 mask_list에 해당하는 train을 여기에 넣어준다. 그리고 Dataloader돌림.
def get_pseudo_labeled_dataloader(pse_table, pse_count, train_labeled_dataset):
    pse_input = []
    pse_key = [key for key, value in pse_count.items() if value >= 1]
    min_value = min(value for value in pse_count.values() if value != 0)

    print("Balance pseudo label class -> ", pse_key, " value -> ", min_value)
    print("add data -> " ,len(pse_key)*min_value)
    
    for i in pse_key:
        matching_tuples = [tup for tup in pse_table if tup[1] == i]
        selected_tuples = random.sample(matching_tuples, min_value)
        pse_input.extend(selected_tuples)


    for i in range(len(pse_input)):
        train_labeled_dataset.add_data(pse_input[i][0],pse_input[i][1])  # 데이터 추가  
    return train_labeled_dataset

def train_one_epoch(netgroup, train_labeled_loader, device):
    netgroup.train()
    total_loss = 0.0
    for batch_label in train_labeled_loader:
        x_lb, y_lb = batch_label['x'], batch_label['label'].to(device)
        outs_x_lb = netgroup.forward(x_lb, y_lb)[0]
        sup_loss_nets = [ce_loss(outs_x_lb, y_lb)]
        netgroup.update(sup_loss_nets)
        total_loss += sup_loss_nets[0].item()
    return total_loss / len(train_labeled_loader)

from criterions.criterions import ce_loss, consistency_loss
def calculate_loss(netgroup, data_loader, device):
    netgroup.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['x'].to(device), batch['label'].to(device)
            outputs = netgroup.forward(inputs, labels)[0]
            loss = [ce_loss(outputs, labels)]
            total_loss += loss[0].item()
    return total_loss / len(data_loader)



# 학습 부분
def oneRun(log_dir, output_dir_experiment, **params):
    
    """ Run one experiment """
    ##### Default Setting #####
    ## Set input data path
    try:
        #data_path = root + 'data/' + params['dataset']
        data_path = params['dataset']
        print('\ndata_path: ', data_path)
    except:
        data_path = root + 'data/ag_news'
        print('\ndata_path is not specified, use default path: ', data_path)

    ## Set output directory: used to store saved model and other outputs
    output_dir_path = output_dir_experiment

    ## Set default hyperparameters
    n_labeled_per_class = 10        if 'n_labeled_per_class' not in params else params['n_labeled_per_class']
    bs = 8                          if 'bs' not in params else params['bs']      # original: 32
    ul_ratio = 10                   if 'ul_ratio' not in params else params['ul_ratio']     
    lr = 2e-5                       if 'lr' not in params else params['lr']      # original: 1e-4, 2e-5  
    lr_linear = 1e-3                if 'lr_linear' not in params else params['lr_linear'] # original: 1e-3      

    # - semi-supervised 
    weight_u_loss =0.1              if 'weight_u_loss' not in params else params['weight_u_loss']
    load_mode = 'semi_SSL'          if 'load_mode' not in params else params['load_mode'] # semi, sup_baseline

    # - pseudo-labeling
    psl_threshold_h = 0.98          if 'psl_threshold_h' not in params else params['psl_threshold_h']   # original: 0.75
    labeling_mode = 'hard'          if 'labeling_mode' not in params else params['labeling_mode'] # hard, soft
    adaptive_threshold = False      if 'adaptive_threshold' not in params else params['adaptive_threshold'] # True, False

    # - ensemble
    num_nets = 1                    if 'num_nets' not in params else params['num_nets'] #SSL을 위한 추가
    cross_labeling = False          if 'cross_labeling' not in params else params['cross_labeling'] # True, False

    # - weight disagreement
    weight_disagreement = False     if 'weight_disagreement' not in params else params['weight_disagreement'] # True, False
    disagree_weight = 1             if 'disagree_weight' not in params else params['disagree_weight']

    # - ema
    ema_mode = False                if 'ema_mode' not in params else params['ema_mode']
    ema_momentum = 0.9              if 'ema_momentum' not in params else params['ema_momentum'] # original: 0.99

    # - others
    seed = params['seed']
    device_idx = 1                  if 'device_idx' not in params else params['device_idx']
    val_interval = 25               if 'val_interval' not in params else params['val_interval'] # 20, 25
    #early_stop_tolerance = 10       if 'early_stop_tolerance' not in params else params['early_stop_tolerance'] # 5, 6, 10
    max_epoch = 10000                if 'max_epoch' not in params else params['max_epoch'] # 100, 200
    max_step = 100000               if 'max_step' not in params else params['max_step'] # 100000, 200000
    
    # Initialize model & optimizer & lr_scheduler
    net_arch = params['net_arch']
    pse_cl = params['pse_cl']

    token = "microsoft/codebert-base" if 'token' not in params else params['token']

    
    # save name
    save_name = 'pls_save_name'       if 'save_name' not in params else params['save_name']       
    
    ## Set random seed and device
    # Fix random seed
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Check & set device
    if torch.cuda.is_available():     
        device = torch.device("cuda", device_idx)
        print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))
    else:
        print('\nNo GPU available, using the CPU instead.')
        device = torch.device("cpu")




    #### Load Data ####
    from utils.dataloader import get_dataloader
    print("\n**line 147 모델 => ",net_arch)
    print('**tokenizer type = ',token,'\n')
    
    print("pesudo label class개수 기준 ->", pse_cl)
    
    if net_arch != token:
        print("net_arch != tokenizer!!!")
        input()
        
    
    #train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes, train_dataset_l, shuffled_train_dataset_u = get_dataloader(data_path, n_labeled_per_class, bs, load_mode)
    train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes, train_dataset_l, shuffled_train_dataset_u, tokenizer = get_dataloader(data_path, n_labeled_per_class, bs, load_mode, token)
    # tokenizer 분석하는 코드 추가함.
    # num = []
    # for _ in range(len(iter(train_labeled_loader))): 
    #     for j in iter(train_labeled_loader).next()['x']['input_ids']:
    #         num.append(len(j))
         
    # for _ in range(len(iter(train_unlabeled_loader))): 
    #     for j in iter(train_labeled_loader).next()['x']['input_ids']:
    #         num.append(len(j))   
    

    print('n_classes: ', n_classes, '\n')


    ##### Model & Optimizer & Learning Rate Scheduler #####
    from models.netgroup import NetGroup
    netgroup = NetGroup(net_arch, num_nets, n_classes, device, lr, lr_linear)
    # Initialize EMA
    netgroup.train()
    if ema_mode:
        netgroup.init_ema(ema_momentum)





    ##### Training & Evaluation #####
    ## Set or import criterions & helper functions
    from utils.helper import format_time
    from criterions.criterions import ce_loss, consistency_loss


    ## Evaluation
    # define evaluation metrics
    # from torchmetrics import F1Score
    from sklearn.metrics import f1_score, accuracy_score
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassConfusionMatrix
    # f1 = F1Score(num_classes=n_classes, average='macro')
    # accuracy = Accuracy(num_classes=n_classes, average='micro')
    accuracy_classwise = Accuracy(num_classes=n_classes, average='none')
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes)


    @torch.no_grad() # no need to track gradients in validation
    def evaluation(loader, final_eval=False):
        """Evaluation"""
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        netgroup.eval()
        if ema_mode: # use ema model for evaluation
            netgroup.eval_ema()
        
        # Tracking variables
        preds_all = []
        target_all = []

        # Evaluate data for one epoch
        for batch in loader:
            b_labels = batch['label'].to(device)

            # forward pass
            outs = netgroup.forward(batch['x'], b_labels)
    
            # take average probs from all nets
            probs = torch.mean(torch.softmax(torch.stack(outs), dim=2), dim=0)

            # Move preds and labels to CPU
            preds = torch.argmax(probs, dim=1)
            target = b_labels

            # For calculating classwise acc
            preds_all.append(preds)
            target_all.append(target)

            #2024-01-25 class별 개수를 구합니다.
            unique, counts = torch.unique(b_labels, return_counts=True)
            class_counts = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
            # Sort the dictionary by keys (class labels)
            class_counts = {k: v for k, v in sorted(class_counts.items())}
            #print(f"Batch class counts: {class_counts}")
            
        # Calculate acc and macro-F1
        # print('preds_all', preds_all)
        # print('target_all', target_all)
        preds_all = torch.cat(preds_all).detach().cpu()
        target_all = torch.cat(target_all).detach().cpu()
       
        

        
        
        
        # print('preds_all', preds_all)
        # print('target_all', target_all)
        
        acc = accuracy_score(target_all, preds_all)
        f1 = f1_score(target_all, preds_all, average='macro')
        #f1_micro = f1_score(target_all, preds_all, average='micro')
        
        # Calculate classwise acc
        accuracy_classwise_ = accuracy_classwise(preds_all, target_all).numpy().round(3)
        
        if final_eval:
            # compute confusion matrix for the final evaluation on the saved model
            confmat_result = confusion_matrix(preds_all, target_all)
            return acc, f1, list(accuracy_classwise_), confmat_result
        else:
            return acc, f1,  list(accuracy_classwise_)




    ## Training
    import time
    import torch.nn.functional as F
    from utils.helper import freematch_fairness_loss

    # Initialize variables
    t0 = time.time() # Measure how long the training takes.
    step = 0
    best_acc = 0
    val_test_acc = 0
    val_test_f1 = 0
    
    best_val_loss = 99
    best_train_loss = 99
    
    best_model_step = 0
    pslt_global = 0
    psl_total_eval = 0
    psl_correct_eval = 0
    cw_psl_total, cw_psl_correct = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_accum, cw_psl_correct_accum = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    training_stats = []
    early_stop_flag = False
    cw_avg_prob = (torch.ones(n_classes) / n_classes).to(device)    # estimate learning stauts of each class
    local_threshold = torch.zeros(n_classes, dtype=int)

    


    from utils.dataloader import MyCollator_SSL, BalancedBatchSampler
    from transformers import AutoTokenizer
    from torch.utils.data import Dataset, DataLoader, Sampler
    # Training
    netgroup.train()
    
    from tqdm import tqdm
    pbar = tqdm(total=max_epoch, desc="Training", position=0, leave=True)
    
    pse_work = False
    best_val_stop = False
    pse_epoch = []
    pse_add_epoch = []
    
    train_losses = []
    val_losses = []


    

    
    for epoch in range(max_epoch): 
        pse_table = []
        if data_path.split('/')[3] == 'corcode_extended_data':
            pse_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}    
        else:
            pse_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
        
        if step > max_step:
            print("조기종료 step > max step =>", step > max_step)
            break
        # if early_stop_flag:
        #     print("acc_val 증가안된 epoch시점 : ", epoch - (early_stop_tolerance))
        #     print("early_stop_flag")
        #     #break
    
        # 결합된 데이터에 대한 DataLoader 생성
        #tokenizer = AutoTokenizer.from_pretrained(token, trust_remote_code=True)
        train_sampler = BalancedBatchSampler(train_dataset_l,bs)
        
        # MyCollator_SSL은 기존의 MyCollator는 data를 augmentation을 함.
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=bs, shuffle= train_sampler, collate_fn=MyCollator_SSL(tokenizer))
        print('\n\n')
        print('line 303 => train data수', len(train_dataset_l))
        print("line 258 => 인스턴스 수" , len(iter(train_labeled_loader)))
        

        acc_test, f1_test, acc_test_cw = evaluation(test_loader)
        acc_val, f1_val, acc_val_cw  = evaluation(dev_loader)
        acc_train, f1_train, acc_train_cw = evaluation(train_labeled_loader)

        if ema_mode:
            netgroup.train_ema()
        netgroup.train()

        #print('>>Epoch %d Step %d acc_test %f f1_test %f acc_val %f f1_val %f acc_train %f f1_train %f psl_cor %d psl_totl %d pslt_global %f '% 
        #        (epoch, step, acc_test, f1_test,acc_val, f1_val, acc_train, f1_train, psl_correct_eval, psl_total_eval, pslt_global),
        #        'Tim {:}'.format(format_time(time.time() - t0)))
        
        print('Step %d '%(step),'Tim {:}'.format(format_time(time.time() - t0)))
        
        
        # Record all statistics from this evaluation.
        acc_psl = (psl_correct_eval/psl_total_eval) if psl_total_eval > 0 else None

        training_stats.append(
            {   'step': step, #배치수
                'acc_train': acc_train,#train의 acc
                'f1_train': f1_train, #train의 f1
                'cw_acc_train': acc_train_cw, #train의 class별 acc
                'acc_val': acc_val,#valid의 acc
                'f1_val': f1_val, #valid의 f1
                'cw_acc_val': acc_val_cw, #valid의 class별 acc
                'acc_test': acc_test,#test의 acc
                'f1_test': f1_test, #test의 f1 macro
                'cw_acc_test': acc_test_cw, #test의 class별 acc
                'psl_correct': psl_correct_eval, # 
                'psl_total': psl_total_eval,
                'acc_psl': acc_psl, 
                'pslt_global': pslt_global,  
                'cw_avg_prob': cw_avg_prob.tolist(),
                'local_threshold': local_threshold.tolist(),
                # 'cw_psl_total': cw_psl_total.tolist(),
                # 'cw_psl_correct': cw_psl_correct.tolist(),  
                'cw_psl_total_eval': cw_psl_total_eval.tolist(),
                'cw_psl_correct_eval': cw_psl_correct_eval.tolist(),
                'cw_psl_acc_eval': (cw_psl_correct_eval/cw_psl_total_eval).tolist(),
                'cw_psl_total_accum': cw_psl_total_accum.tolist(),
                'cw_psl_correct_accum': cw_psl_correct_accum.tolist(),
                'cw_psl_acc_accum': (cw_psl_correct_accum/cw_psl_total_accum).tolist(),
            })

        # check classwise psl accuracy and total psl accuracy for the current eval
        print('acc_train_cw(현재 train의 class별 acc)',acc_train_cw)
        print('cw_psl_total_eval(pseudo label 클래스별 총 샘플 수): ', cw_psl_total_eval.tolist())
        print('cw_psl_correct_eval(pseudo label 클래스별 맞은 샘플 수): ', cw_psl_correct_eval.tolist())
        #print('class-wise-accuracy -> ',  cw_psl_correct_eval.tolist() / cw_psl_total_eval.tolist())
        print('psl_acc(PSL 평가에서의 정확도): ', round((psl_correct_eval/psl_total_eval),3), end=' ') if psl_total_eval > 0 else print('psl_acc(PSL 평가에서의 정확도): None', end=' ')
        print('\ncw_psl_acc(클래스별 PSL 평가에서의 정확도): ', (cw_psl_correct_eval/cw_psl_total_eval).tolist())


        # if epoch+1 == 100:
        #     data['epoch99기준 train_class별 acc'].append(acc_train_cw)
        #     data['epoch 99기준 예측 class'].append(cw_psl_total_eval.tolist())


        # Early stopping & Save best model
        # - best criterion: acc_val 
        # 검증 val보다 best acc가 높아서 조기 종료가 됨.
        if acc_val > best_acc:
            best_acc = acc_val
            best_model_step = step
            
            best_epoch = epoch
            
            val_test_acc = acc_test
            val_test_f1 = f1_test
            
            
            #early_stop_count = 0
            if best_val_stop == False:
                print(f"***best_acc_model_save --- epoch = {epoch+1}***")
                netgroup.save_model(output_dir_path, save_name, ema_mode=ema_mode)
                
        if pse_work == True:
            print(f"***pse_model_save --- epoch = {epoch+1}***")
            netgroup.save_model(output_dir_path, save_name, ema_mode=ema_mode)
            pse_work = False
        # else:
        #     early_stop_count+=1
        #     if early_stop_count >= early_stop_tolerance:
        #         early_stop_flag = True
        #         print("**조기종료됨**\n")
        #         break

        # initialize pseudo labels evaluation
        psl_total_eval, psl_correct_eval = 0, 0
        #psl_correct_eval = 0
        cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)




        step += len(iter(train_labeled_loader))

                
        


        # Process Unlabeled Data
        if load_mode == 'semi_SSL':
            ul_ratio = len(train_unlabeled_loader)

            for ratio in range(ul_ratio):
                try:
                    batch_unlabel = next(data_iter_unl)

                #except StopIteration:
                except :
                    data_iter_unl = iter(train_unlabeled_loader)
                    batch_unlabel = next(data_iter_unl) # data_iter_unl은 7개의 ul_data가 잇음
                
                # unlabel data 가져오기
                x_ulb_s = batch_unlabel['x']


                # Forward pass for the self-supervised task using weak augmentation
                # SSL 체크부분
                with torch.no_grad():
                    # unlabel data의 예측값
                    
                    # symbolic-based pseudo-label 모듈 사용할 부분 .forward 대신에 netgroup.py에서 symbolic-based pseudo-label 모듈 호출후 process_code이용하면 될거라고 생각함.#################################### 
                    
                    #decoder_input_ids = x_ulb_s.to(device)
                    #batch_unlabel['label'].to(device)
                    
                    outs_x_ulb_w_nets = netgroup.forward(x_ulb_s)


                # Generate pseudo labels and masks for all nets in one batch of unlabeled data
                pseudo_labels_nets = []
                u_psl_masks_nets = []
                
            
                for i in range(num_nets):
                    # Generate pseudo labels
                    logits_x_ulb_w = outs_x_ulb_w_nets[i]
                    if labeling_mode == 'soft':
                        pseudo_labels_nets.append(torch.softmax(logits_x_ulb_w, dim=-1))
                    else:
                        max_idx = torch.argmax(logits_x_ulb_w, dim=-1)
                        pseudo_labels_nets.append(F.one_hot(max_idx, num_classes=n_classes).to(device))


  
                    # Compute mask for pseudo labels
                    probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                    


                    
                    max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
       
                    if ratio < 5:
                        print('**pseudo_label -> ', [round(prob, 4) for prob in probs_x_ulb_w.squeeze().tolist()], "**")
                        print("max = ", round(max_probs.squeeze().tolist(),4), "| max_idx = ", max_idx.squeeze().tolist())

                        
                    if adaptive_threshold:
                        # Fixed hard threshold
                        pslt_global = psl_threshold_h
                        u_psl_masks_nets.append(max_probs >= pslt_global)
                    
                    else:      
                        print("threshold error")
                        input()              
                    
                    # else:
                    #     # Adaptive local threshold
                    #     pslt_global = psl_threshold_h
                    #     cw_avg_prob = cw_avg_prob * ema_momentum + torch.mean(probs_x_ulb_w, dim=0) * (1 - ema_momentum)
                    #     local_threshold = cw_avg_prob / torch.max(cw_avg_prob, dim=-1)[0]
                    #     u_psl_mask = max_probs.ge(pslt_global * local_threshold[max_idx])
                    #     u_psl_masks_nets.append(u_psl_mask)

                # Compute loss for unlabeled data for all nets
                #total_unsup_loss_nets = []
                
                # 수정필요
                if any(any(item) for item in u_psl_masks_nets):
                    pse_data = shuffled_train_dataset_u[ratio][0]
                    pse_label = max_idx[0].item()+1
                    pse_count[pse_label] += 1
                    pse_table.append((pse_data, pse_label))
                    

                if ratio == (ul_ratio-1):   
                    pse_class = sum(1 for value in pse_count.values() if value >= 1)
                    
                    if pse_class > 0:
                        pse_epoch.append(epoch+1)

                    #print('\npse_class -> ', pse_class)
                    if pse_class >= int(pse_cl):
                        pse_add_epoch.append(epoch+1)                                          
                        train_dataset_l =  get_pseudo_labeled_dataloader(pse_table, pse_count, train_dataset_l)
                        pse_work = True
                        best_val_stop = True
                    #else :
                    #    print('\n')
                    
                

                pseudo_label = pseudo_labels_nets[0]
                u_psl_mask = u_psl_masks_nets[0]

                # for i in range(num_nets):
                    #pseudo_label = pseudo_labels_nets[i]
                    #u_psl_mask = u_psl_masks_nets[i]
                    
                    #if weight_disagreement:
                        #disagree_mask = torch.logical_xor(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        #agree_mask = torch.logical_and(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        #disagree_weight_masked = disagree_weight * disagree_mask + (1 - disagree_weight) * agree_mask
                    #elif weight_disagreement == 'ablation_baseline':
                        #disagree_weight = 0.5
                        #disagree_mask = torch.logical_xor(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        #agree_mask = torch.logical_and(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        #disagree_weight_masked = disagree_weight * disagree_mask + (1 - disagree_weight) * agree_mask
                    #else:
                        #disagree_weight_masked = None

                    # Compute loss for unlabeled data
                    #unsup_loss = consistency_loss(outs_x_ulb_w_nets[i], pseudo_label, loss_type='ce', mask=u_psl_mask, disagree_weight_masked=disagree_weight_masked)
                    #total_unsup_loss = weight_u_loss * unsup_loss
                    #total_unsup_loss_nets.append(total_unsup_loss)                




                # Update netgroup from loss of unlabeled data
                # netgroup.update(total_unsup_loss_nets)
                if ema_mode:
                    netgroup.update_ema()

                # Additional Evaluation Metrics
                
                
                gt_labels_u = batch_unlabel['label'][u_psl_mask].to(device)
                psl_total = torch.sum(u_psl_mask).item()

                u_label_psl = pseudo_label[u_psl_mask]
                u_label_psl_hard = torch.argmax(u_label_psl, dim=-1)
                
                # SSL 확인해야함.###############################################################
                psl_correct = torch.sum(u_label_psl_hard == gt_labels_u).item()
                #print("psl_correct = ", psl_correct)

                psl_total_eval += psl_total
                psl_correct_eval += psl_correct

                # Class-wise total and correct number of pseudo-labels
                cw_psl_total = torch.bincount(u_label_psl_hard, minlength=n_classes).to('cpu')
                cw_psl_correct = torch.bincount(u_label_psl_hard[u_label_psl_hard == gt_labels_u], minlength=n_classes).to('cpu')

                cw_psl_total_eval += cw_psl_total
                cw_psl_correct_eval += cw_psl_correct

                cw_psl_total_accum += cw_psl_total
                cw_psl_correct_accum += cw_psl_correct
        
        
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], shuffle=True, collate_fn=MyCollator_SSL(tokenizer))
        train_loss = train_one_epoch(netgroup, train_labeled_loader, device)   
        val_loss = calculate_loss(netgroup, dev_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_epoch = epoch
            
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            train_epoch = epoch
        
        if epoch == 0:
            first_train_loss = train_loss
            first_val_loss = val_loss
           
        if epoch+1 == max_epoch:
            last_train_loss = train_loss
            last_val_loss = val_loss

            
            
        pbar.write(f"Epoch {epoch + 1}/{max_epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                   f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1(macro): {f1_test:.4f}, "
                   f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}, "
                   f"Train Data Number: {len(train_dataset_l)}")
        pbar.update(1)
    pbar.close()
                
                



                    


    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t0)))




    ##### Results Summary and Visualization
    # Load the saved best models and evaluate on the test set
    netgroup.load_model(output_dir_path, save_name, ema_mode=ema_mode)
    acc_test, f1_test, acc_test_cw, confmat_test = evaluation(test_loader, final_eval=True)


    ## Quantatitive Results
    # Save training statistics
    pd.set_option('precision', 4)
    df_stats= pd.DataFrame(training_stats)
    # df_stats = df_stats.set_index('step')
    if 'step' not in df_stats.columns:
        df_stats['step'] = range(1, len(df_stats) + 1)
    df_stats = df_stats.set_index('step')
    training_stats_path = log_dir + 'training_statistics.csv'   
    df_stats.to_csv(training_stats_path)     
    print('Save training statistics in: ', training_stats_path)

    # Save best record in both the training statisitcs and summary file
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    best_data = {'record_time': cur_time,
                'best_step': best_model_step, 'test_acc':acc_test, 'test_f1': f1_test     
                }
    
    
    
    
    # data['best_train_loss'].append(best_train_loss)
    # data['epoch'].append(train_epoch+1)    
    # data['train_loss(1epoch)'].append(first_train_loss)
    # data['best_val_loss'].append(best_val_loss)
    # data['val_epoch'].append(val_epoch+1)  
    # data['val_loss(1epoch)'].append(first_val_loss)
    
    
    print('\nbest_trian_loss, epoch =>', best_train_loss, train_epoch+1)
    print("Epoch-1_train_loss =>", first_train_loss)
    print("max_epoch_train_loss =>", last_train_loss, '\n')
    
    
    
    print('\nbest_val_loss, epoch => ', best_val_loss, val_epoch+1)
    print("Epoch-1_val_loss =>", first_val_loss)
    print("max_epoch_val_loss =>", last_val_loss, '\n')
    
 
    
    print("pse_epoch => ", pse_epoch)
    print("추가되는 pse_epoch => ", pse_add_epoch,'\n')
    
    print('total_data: ', len(train_dataset_l),'\n\nBest_step: ', best_model_step,'\nBest_val_epoch: ', best_epoch+1 ,
          '\nbest_val_acc: ',best_acc, '\nbest_val_test_acc: ', val_test_acc, '\nbest_val_test_f1: ', val_test_f1)
    

    best_data.update(params) # record tuned hyper-params
    best_df = pd.DataFrame([best_data])         
    best_csv_path = log_dir_multiRun + 'summary.csv'
    if not os.path.exists(best_csv_path):
        best_df.to_csv(best_csv_path, mode='a', index=False, header=True)
    else:
        best_df.to_csv(best_csv_path, mode='a', index=False, header=False)
    best_df.to_csv(training_stats_path, mode='a', index=False, header=True)
    print('Save best record in: ', best_csv_path, end='')
     
    # update_csv(data, csv_path)   



    # Visualization - Plot Training Curves
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select data range and types to plot
    df_stats_1 = df_stats
    plot_types = ['f1', 'acc', 'psl', 'pslt']   # ['f1', 'acc', 'psl', 'pslt']

    for plot_type in plot_types:
        plt.figure()
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        # plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        for idx, key in enumerate(df_stats.keys().tolist()):
            if key.split('_')[0] == plot_type:
                plt.plot(df_stats_1[key], '--', label=key)
        # Label the plot.
        plt.xlabel("iteration")
        plt.ylabel("peformance")
        plt.legend()
        plt.savefig(log_dir+plot_type+'.png', bbox_inches='tight')

    # loss 값의 변화를 그래프로 시각화합니다.
    plt.figure(figsize=(20,14))
    #epochs = range(1, max_epoch + 1)  # epoch 수
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(log_dir+'loss_plot.png', bbox_inches='tight')
        
    # Visualize and save confusion matrix
    df_cm = pd.DataFrame(confmat_test, index = [i for i in range(n_classes)],
                    columns = [i for i in range(n_classes)], dtype=int)
    df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(20,14))
    sns.heatmap(df_cm_norm, annot=True, annot_kws={"size": 10}, fmt='.2f', cmap='Reds')
    plt.savefig(log_dir+'confmat_norm.png', bbox_inches='tight')
    # df_cm.to_csv(log_dir+'confmat.csv', index=True, header=True)





    # return best_data to record multiRun results
    return best_data



#출력하는 부분, 학습 횟수
##### multiRun ######
def multiRun(experiment_home=None, num_runs=3, unit_test_mode=False, **params):
    import statistics
    import pandas as pd
    import os
    import random
    import time

    ## Create folder structure
    # genereate a list of fixed seeds according to the number of runs
    #seeds_list = list(range(num_runs)) if 'seeds_list' not in params else params['seeds_list']
    seeds_list = params['seeds_list']
    
    
    # create a folder for this multiRun
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    if experiment_home is None:
        experiment_home = root + '/experiment/temp/'
    log_root = experiment_home + '/log/'
    global log_dir_multiRun
    log_dir_multiRun = log_root + f"{params['model_name']}/class{params['pse_cl']}" + '/'

    if not os.path.exists(experiment_home):
        os.makedirs(experiment_home)
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(log_dir_multiRun):
        os.makedirs(log_dir_multiRun)

    # create folders for each run
    for i in range(num_runs):
        #log_dir = log_dir_multiRun + str(seeds_list[i]) + '/' + params['save_name']
        log_dir = log_dir_multiRun + '/' + params['save_name']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    # create output folder for this experiment
    output_dir_experiment = experiment_home + '/output/'
    if not os.path.exists(output_dir_experiment):
        os.makedirs(output_dir_experiment)

    
    ## Obtain averaged results over multiple runs
    results = []
    test_accs = []
    test_f1s = []
    
    for i in range(num_runs):
        if not unit_test_mode:
            #log_dir = log_dir_multiRun + str(seeds_list[i]) + '/'
            log_dir = log_dir_multiRun + '/' + params['save_name'] + '/'
            result = oneRun(log_dir, output_dir_experiment, **params, seed=seeds_list[i])
        else:
            result = {'test_acc': random.random(), 'test_f1': random.random()}
        results.append(result)
        
        test_accs.append(result['test_acc'])
        test_f1s.append(result['test_f1'])

    st_dev_acc = round(statistics.pstdev(test_accs), 4)
    mean_acc = round(statistics.mean(test_accs), 4)
    st_dev_f1 = round(statistics.pstdev(test_f1s), 4)
    mean_f1 = round(statistics.mean(test_f1s), 4)

    
    if len(seeds_list) != 1:
        final = {'record_time': cur_time,
                'Mean_std_acc': '%.2f ± %.2f' % (100*mean_acc, 100*st_dev_acc),
                'Mean_std_f1': '%.2f ± %.2f' % (100*mean_f1, 100*st_dev_f1)}
    else:
        final = {'record_time': cur_time,
                'Mean_std_acc': '%.2f' % (100*mean_acc),
                'Mean_std_f1': '%.2f' % (100*mean_f1)}

    final.update(params)
    final.update({'seeds_list': seeds_list})
    df = pd.DataFrame([final])
    csv_path = log_root + 'summary_avgrun.csv'
    df.to_csv(csv_path, mode='a', index=False, header=True)
    print('\nSave best record in: ', csv_path)