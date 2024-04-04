import os
import random
import sys
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import preprocessor as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix

from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

from utils.dataloader import get_dataloader
from models.netgroup import NetGroup
from utils.helper import format_time
from criterions.criterions import ce_loss, consistency_loss
from utils.helper import freematch_fairness_loss
from utils.dataloader import MyCollator_SSL, BalancedBatchSampler


#### Set Path ####
# go to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = '../' #origin/
root_code = '../code' #origin/code
sys.path.append(root)
sys.path.append(root_code)
print('current work directory: ', os.getcwd())



def get_pseudo_labeled_dataloader(train_labeled_dataset, ul_data, ul_list, max_idx, index, bs):
    if len(ul_list[0]) != bs:
        print("**bs에 맞지않음!!!!!**\n")
        return train_labeled_dataset 
    batch_unlabeled = []
    for batch_index in range(index*bs, index*bs+bs):
        batch_unlabeled.append(ul_data[batch_index])

    total = len(train_labeled_dataset)

    for i in range(0, bs):
        mask = ul_list[0][i]
        idx = max_idx[i]
        if mask.item():
            train_labeled_dataset.add_data(batch_unlabeled[i][0],idx.item()+1)
    return train_labeled_dataset

def oneRun(output_dir_path, **params):
    
    """ Run one experiment """
    ## Set input data path
    try:
        data_path = root + 'data/' + params['dataset']
        #data_path = params['datase']
        print('\ndata_path: ', data_path)
    except:
        data_path = root + 'data/ag_news'
        print('\ndata_path is not specified, use default path: ', data_path)

    ## Set output directory: used to store saved model and other outputs
    #output_dir_path = experiment_home

    ## Set default hyperparameters
    n_labeled_per_class = params['n_labeled_per_class']
    bs = params['bs']
    # ul_ratio = params['ul_ratio']     
    psl_threshold_h = params['psl_threshold_h']
    lr = params['lr']
    #lr_linear = params['lr_linear']

    weight_u_loss = params['weight_u_loss']
    load_mode = params['load_mode'] # semi, sup_baseline

    labeling_mode = params['labeling_mode'] # hard, soft
    adaptive_threshold = params['adaptive_threshold'] # True, False

    # - ensemble
    num_nets = params['num_nets'] #SSL을 위한 추가
    cross_labeling = params['cross_labeling'] # True, False

    # - weight disagreement
    weight_disagreement = params['weight_disagreement'] # True, False
    disagree_weight = params['disagree_weight']

    # - ema
    ema_mode = params['ema_mode']
    ema_momentum = params['ema_momentum'] # original: 0.99

    # - others
    seed = params['seed']
    device_idx = params['device_idx']
    val_interval = params['val_interval'] # 20, 25
    early_stop_tolerance = params['early_stop_tolerance'] # 5, 6, 10
    max_epoch = params['max_epoch'] # 100, 200
    max_step = params['max_step'] # 100000, 200000
    
    # Initialize model & optimizer & lr_scheduler
    net_arch = params['net_arch']

    
    # save name
    save_name = params['save_name']       
    
    ## Set random seed and device
    # Fix random seed
    cudnn.deterministic = True
    cudnn.benchmark = True
    

    # Check & set device
    assert torch.cuda.is_available()
    device = torch.device("cuda", device_idx)
    print('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU-', device_idx, torch.cuda.get_device_name(device_idx))

    tokenizer = AutoTokenizer.from_pretrained(net_arch)


    #### Load Data ####
    print("\n**line 147 모델 => ",net_arch)
    print('**tokenizer type = ',net_arch,'\n')
    
    #train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes, train_dataset_l, shuffled_train_dataset_u = get_dataloader(data_path, n_labeled_per_class, bs, load_mode)
    train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes, train_dataset_l, shuffled_train_dataset_u = get_dataloader(data_path, n_labeled_per_class, bs, load_mode, net_arch)
   
    print('n_classes: ', n_classes, '\n')
    #print('line 152')
    # # used for degugging
    # sys.exit()




    ##### Model & Optimizer & Learning Rate Scheduler #####
    # netgroup = NetGroup(net_arch, num_nets, n_classes, device, lr, lr_linear)
    netgroup = NetGroup(net_arch, num_nets, n_classes, device, lr)
    
    # Initialize EMA
    netgroup.train()
    if ema_mode:
        netgroup.init_ema(ema_momentum)





    ##### Training & Evaluation #####
    ## Set or import criterions & helper functions


    ## Evaluation
    # define evaluation metrics
    # from torchmetrics import F1Score
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
        
        # Calculate classwise acc
        accuracy_classwise_ = accuracy_classwise(preds_all, target_all).numpy().round(3)

        if final_eval:
            # compute confusion matrix for the final evaluation on the saved model
            confmat_result = confusion_matrix(preds_all, target_all)
            return acc, f1, list(accuracy_classwise_), confmat_result
        else:
            return acc, f1, list(accuracy_classwise_)



    ## Training

    # Initialize variables
    t0 = time.time() # Measure how long the training takes.
    step = 0
    best_acc = 0
    best_model_step = 0
    pslt_global = 0
    psl_total_eval = 0
    psl_correct_eval = 0
    early_stop_count = 0
    cw_psl_total, cw_psl_correct = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_accum, cw_psl_correct_accum = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    training_stats = []
    early_stop_flag = False
    cw_avg_prob = (torch.ones(n_classes) / n_classes).to(device)    # estimate learning stauts of each class
    local_threshold = torch.zeros(n_classes, dtype=int)




    # Training
    pbar = tqdm(total=max_step,desc="{} training".format(net_arch))
    netgroup.train()
    for epoch in range(max_epoch):
        epoch += 1
        if step > max_step:
            print("조기종료 step > max step =>", step > max_step)
            break
        if early_stop_flag:
            print("종료된 epoch시점 : ", epoch, '(epoch - 11에서부터 acc_val증가 안됨.)')
            print("조기종료 early_stop_flag ")
            break
        
        k = epoch #epoch마다 print를 위해서 대입
        
        
        # 결합된 데이터에 대한 DataLoader 생성
        train_sampler = BalancedBatchSampler(train_dataset_l,bs)
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=bs, shuffle= train_sampler, collate_fn=MyCollator_SSL(tokenizer))
        
        print('\nline 303 => train data수', len(train_dataset_l))
        print("line 258 => 인스턴스 수" , len(iter(train_labeled_loader)))


        for batch_label in iter(train_labeled_loader):
            # --- Evaluation: Check Performance on Validation set every val_interval batches ---##
            if epoch == k :
                k += 1      
                print("=======test_loader=======")
                acc_test, f1_test, acc_test_cw = evaluation(test_loader)
                print("=======dev_loader=======")
                acc_val, f1_val, acc_val_cw = evaluation(dev_loader)
                print("=======train_labeled_loader=======")
                acc_train, f1_train, acc_train_cw = evaluation(train_labeled_loader)
                print("=======finish=======")
                # print('acc_train_cw:',acc_train_cw)
                # evaluation(train_labeled_loader)
                # exit()
                # restore training mode 
                if ema_mode:
                    netgroup.train_ema()
                netgroup.train()

                print('>>Epoch %d Step %d acc_test %f f1_test %f acc_val %f f1_val %f acc_train %f f1_train %f psl_cor %d psl_totl %d pslt_global %f '% 
                        (epoch, step, acc_test, f1_test,acc_val, f1_val, acc_train, f1_train, psl_correct_eval, psl_total_eval, pslt_global),
                        'Tim {:}'.format(format_time(time.time() - t0)))
                
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
                        'f1_test': f1_test, #test의 f1
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
                print('acc_train_cw',acc_train_cw)
                print('cw_psl_eval: ', cw_psl_total_eval.tolist(), cw_psl_correct_eval.tolist())
                print('psl_acc: ', round((psl_correct_eval/psl_total_eval),3), end=' ') if psl_total_eval > 0 else print('psl_acc: None', end=' ')
                print('cw_psl_acc: ', (cw_psl_correct_eval/cw_psl_total_eval).tolist())

                # Early stopping & Save best model
                # - best criterion: acc_val 
                # 검증 val보다 best acc가 높아서 조기 종료가 됨.
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_model_step = step
                    early_stop_count = 0
                    netgroup.save_model(output_dir_path, save_name, ema_mode=ema_mode)
                else:
                    early_stop_count+=1
                    if early_stop_count >= early_stop_tolerance:
                        early_stop_flag = True
                        print('Early stopping trigger at step: ', step)
                        print('Best model at step: ', best_model_step)
                        print("**조기종료됨**")
                        break

                # initialize pseudo labels evaluation
                psl_total_eval, psl_correct_eval = 0, 0
                #psl_correct_eval = 0
                cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)






            ## --- Training --- ##
            step += 1
            ## Process Labeled Data
            #x_lb, y_lb = batch_label['x_w'], batch_label['label']
            x_lb, y_lb = batch_label['x'], batch_label['label']
            
            if len(y_lb) == bs :
                # forward pass
                outs_x_lb = netgroup.forward(x_lb, y_lb.to(device))
                # compute loss for labeled data
                sup_loss_nets = [ce_loss(outs_x_lb[i], y_lb.to(device)) for i in range(num_nets)]
                # update netgorup from loss of labeled data
                netgroup.update(sup_loss_nets)
                if ema_mode:
                    netgroup.update_ema()
                    
            
            else : 
                print('train batch instace수가 7안됨 pass')
                    
                
        



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
                    outs_x_ulb_w_nets = netgroup.forward(x_ulb_s)

                # Generate pseudo labels and masks for all nets in one batch of unlabeled data
                pseudo_labels_nets = []
                u_psl_masks_nets = []
                
                
                import pdb; pdb.set_trace

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
                    #print(max_probs)
                    #print(max_idx)
                    
                    if not adaptive_threshold:
                        # Fixed hard threshold
                        pslt_global = psl_threshold_h
                        u_psl_masks_nets.append(max_probs >= pslt_global)                    
                    
                    else:
                        # Adaptive local threshold
                        pslt_global = psl_threshold_h
                        cw_avg_prob = cw_avg_prob * ema_momentum + torch.mean(probs_x_ulb_w, dim=0) * (1 - ema_momentum)
                        local_threshold = cw_avg_prob / torch.max(cw_avg_prob, dim=-1)[0]
                        u_psl_mask = max_probs.ge(pslt_global * local_threshold[max_idx])
                        u_psl_masks_nets.append(u_psl_mask)

                # Compute loss for unlabeled data for all nets
                total_unsup_loss_nets = []
                
                if any(any(item) for item in u_psl_masks_nets):
                    train_dataset_l =  get_pseudo_labeled_dataloader(train_dataset_l, shuffled_train_dataset_u, u_psl_masks_nets, max_idx, ratio, bs)

                    
                


                for i in range(num_nets):
                    pseudo_label = pseudo_labels_nets[i]
                    u_psl_mask = u_psl_masks_nets[i]
                    
                    if weight_disagreement:
                        disagree_mask = torch.logical_xor(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        agree_mask = torch.logical_and(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        disagree_weight_masked = disagree_weight * disagree_mask + (1 - disagree_weight) * agree_mask
                    elif weight_disagreement == 'ablation_baseline':
                        disagree_weight = 0.5
                        disagree_mask = torch.logical_xor(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        agree_mask = torch.logical_and(u_psl_masks_nets[(i) % num_nets], u_psl_masks_nets[(i + 1) % num_nets])
                        disagree_weight_masked = disagree_weight * disagree_mask + (1 - disagree_weight) * agree_mask
                    else:
                        disagree_weight_masked = None

                    # Compute loss for unlabeled data
                    unsup_loss = consistency_loss(outs_x_ulb_w_nets[i], pseudo_label, loss_type='ce', mask=u_psl_mask, disagree_weight_masked=disagree_weight_masked)
                    total_unsup_loss = weight_u_loss * unsup_loss
                    total_unsup_loss_nets.append(total_unsup_loss)                




                # Update netgroup from loss of unlabeled data
                # netgroup.update(total_unsup_loss_nets)
                if ema_mode:
                    netgroup.update_ema()

                # Additional Evaluation Metrics
                
                
                gt_labels_u = batch_unlabel['label'][u_psl_mask].to(device)
                #print("line 442, gt_labels_u:{}".format(gt_labels_u))
                
                psl_total = torch.sum(u_psl_mask).item()
                #print("psl_total =", psl_total)

                u_label_psl = pseudo_label[u_psl_mask]
                #print("line 448, u_label_psl:{}".format(u_label_psl))
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

        pbar.update(1)
            



                    


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
    training_stats_path = experiment_home + 'training_statistics.csv'   
    df_stats.to_csv(training_stats_path)     
    print('Save training statistics in: ', training_stats_path)

    # Save best record in both the training statisitcs and summary file
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    best_data = {'record_time': cur_time,
                'best_step': best_model_step, 'test_acc':acc_test, 'test_f1': f1_test,     
                }
    print('\nBest_step: ', best_model_step, '\nbest_test_acc: ', acc_test, '\nbest_test_f1: ', f1_test)

    best_data.update(params) # record tuned hyper-params
    best_df = pd.DataFrame([best_data])         
    best_csv_path = experiment_home + 'summary.csv'
    if not os.path.exists(best_csv_path):
        best_df.to_csv(best_csv_path, mode='a', index=False, header=True)
    else:
        best_df.to_csv(best_csv_path, mode='a', index=False, header=False)
    best_df.to_csv(training_stats_path, mode='a', index=False, header=True)
    print('Save best record in: ', best_csv_path, end='')
        



    ## Visualization - Plot Training Curves
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
        plt.savefig(experiment_home+plot_type+'.png', bbox_inches='tight')

    # Visualize and save confusion matrix
    df_cm = pd.DataFrame(confmat_test, index = [i for i in range(n_classes)],
                    columns = [i for i in range(n_classes)], dtype=int)
    df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(20,14))
    sns.heatmap(df_cm_norm, annot=True, annot_kws={"size": 10}, fmt='.2f', cmap='Reds')
    plt.savefig(experiment_home+'confmat_norm.png', bbox_inches='tight')
    df_cm.to_csv(experiment_home+'confmat.csv', index=True, header=True)

    # return best_data to record multiRun results
    return best_data



#출력하는 부분, 학습 횟수
##### multiRun ######
def multiRun(unit_test_mode=False, **params):
    import statistics
    import pandas as pd
    import os
    import random
    import time

    ## Create folder structure
    # genereate a list of fixed seeds according to the number of runs
    # seeds_list = list(range(num_runs)) if 'seeds_list' not in params else params['seeds_list']
    experiment_home = params['experiment_home']
    num_runs = params['num_runs']
    seeds_list = params['seeds_list']

    # create a folder for this multiRun
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    if experiment_home is None:
        experiment_home = './experiment/temp/'

    if not os.path.exists(experiment_home):
        os.makedirs(experiment_home)

    # create folders for each run
    #for i in range(num_runs):
        #if not os.path.exists(something_here!!!):
        #    os.makedirs(something_here)


    ## Obtain averaged results over multiple runs
    print(f"Experiment home: {experiment_home}")
    print(f"The number of runs: {num_runs}")
    results = []
    test_accs = []
    test_f1s = []
    for i in range(num_runs):
        if not unit_test_mode:
            print(f"Seed for this run: {seeds_list[i]}")
            result = oneRun(experiment_home, **params, seed=seeds_list[i])
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
    print(final)
    print('\nSave best record in: ', csv_path)
