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
root = '../'
root_code = '../code'
sys.path.append(root)
sys.path.append(root_code)
print('current work directory: ', os.getcwd())


def oneRun(log_dir, output_dir_experiment, **params):
    """ Run one experiment """
    ##### Default Setting #####
    ## Set input data path
    try:
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
    load_mode = 'semi'              if 'load_mode' not in params else params['load_mode'] # semi, sup_baseline

    # - pseudo-labeling
    psl_threshold_h = 0.98          if 'psl_threshold_h' not in params else params['psl_threshold_h']   # original: 0.75
    labeling_mode = 'hard'          if 'labeling_mode' not in params else params['labeling_mode'] # hard, soft
    adaptive_threshold = False      if 'adaptive_threshold' not in params else params['adaptive_threshold'] # True, False

    # - ensemble
    num_nets = 2                    if 'num_nets' not in params else params['num_nets']
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
    early_stop_tolerance = 10       if 'early_stop_tolerance' not in params else params['early_stop_tolerance'] # 5, 6, 10
    max_epoch = 10000                if 'max_epoch' not in params else params['max_epoch'] # 100, 200
    max_step = 100000               if 'max_step' not in params else params['max_step'] # 100000, 200000

    # 추가
    net_arch = params['net_arch']
    token = "microsoft/codebert-base" if 'token' not in params else params['token']
    save_name = 'pls_save_name'       if 'save_name' not in params else params['save_name']  

    ## Set random seed and device
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
    from utils.dataloader_original import get_dataloader
    print("\n**line 107 모델 => ",net_arch)
    print('**tokenizer type = ',token,'\n')
    
    # 각 모델에 대한 데이터 로더 생성
    train_labeled_loaders = []
    train_unlabeled_loaders = []
    dev_loaders = []
    test_loaders = []

    for token in token:
        train_labeled_loader, train_unlabeled_loader, dev_loader, test_loader, n_classes = get_dataloader(data_path, n_labeled_per_class, bs, load_mode, token)
        train_labeled_loaders.append(train_labeled_loader)
        train_unlabeled_loaders.append(train_unlabeled_loader)
        dev_loaders.append(dev_loader)
        test_loaders.append(test_loader)
    print('n_classes: ', n_classes)
    
    
    
    ##### Model & Optimizer & Learning Rate Scheduler #####
    from models.netgroup_original import NetGroup

    # Initialize models & optimizers
    #netgroups = [NetGroup(net_arch, num_nets, n_classes, device, lr, lr_linear) for net_arch in net_archs]
    netgroup = NetGroup(net_arch, num_nets, n_classes, device, lr, lr_linear)
    
    #optimizers = [torch.optim.Adam(netgroup.parameters(), lr=lr) for netgroup in netgroups]

    # Initialize EMA
    for netgroup in netgroups:
        netgroup.train()
        if ema_mode:
            netgroup.init_ema(ema_momentum)

    ##### Training & Evaluation #####
    from utils.helper import format_time
    from criterions.criterions import ce_loss, consistency_loss
    from sklearn.metrics import f1_score, accuracy_score
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassConfusionMatrix

    accuracy_classwise = Accuracy(num_classes=n_classes, average='none')
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes)

    @torch.no_grad()
    def evaluation(loader, final_eval=False):
        """Evaluation"""
        preds_all = []
        target_all = []

        for netgroup in netgroups:
            netgroup.eval()
            if ema_mode:
                netgroup.eval_ema()

        for batch in loader:
            b_labels = batch['label'].to(device)
            all_probs = []

            for netgroup in netgroups:
                outs = netgroup.forward(batch['x'], b_labels)
                probs = torch.mean(torch.softmax(torch.stack(outs), dim=2), dim=0)
                all_probs.append(probs)

            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            preds = torch.argmax(avg_probs, dim=1)
            target = b_labels

            preds_all.append(preds)
            target_all.append(target)

        preds_all = torch.cat(preds_all).detach().cpu()
        target_all = torch.cat(target_all).detach().cpu()
        acc = accuracy_score(target_all, preds_all)
        f1 = f1_score(target_all, preds_all, average='macro')
        accuracy_classwise_ = accuracy_classwise(preds_all, target_all).numpy().round(3)

        if final_eval:
            confmat_result = confusion_matrix(preds_all, target_all)
            return acc, f1, list(accuracy_classwise_), confmat_result
        else:
            return acc, f1, list(accuracy_classwise_)

    import time
    from tqdm import tqdm

    t0 = time.time()
    step = 0
    best_acc = 0
    best_model_step = 0
    pslt_global = 0
    psl_total_eval = 0
    psl_correct_eval = 0
    cw_psl_total, cw_psl_correct = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    cw_psl_total_accum, cw_psl_correct_accum = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)
    training_stats = []
    early_stop_flag = False
    cw_avg_prob = (torch.ones(n_classes) / n_classes).to(device)
    local_threshold = torch.zeros(n_classes, dtype=int)

    pbar = tqdm(total=max_epoch, desc="Training", position=0, leave=True)

    for epoch in range(max_epoch):
        pse_table = []
        pse_count = {i: 0 for i in range(1, n_classes + 1)}

        if step > max_step:
            print("조기종료 step > max step =>", step > max_step)
            break

        for batch_label in zip(*train_labeled_loaders):  # 각 모델에 대해 배치를 병합
            acc_test, f1_test, acc_test_cw = evaluation(test_loaders[0])
            acc_val, f1_val, acc_val_cw = evaluation(dev_loaders[0])
            acc_train, f1_train, acc_train_cw = evaluation(train_labeled_loaders[0])

            for netgroup in netgroups:
                if ema_mode:
                    netgroup.train_ema()
                netgroup.train()

            acc_psl = (psl_correct_eval / psl_total_eval) if psl_total_eval > 0 else None

            training_stats.append(
                {'step': step, 'acc_val': acc_val, 'f1_val': f1_val, 'acc_train': acc_train, 'f1_train': f1_train,
                 'psl_correct': psl_correct_eval, 'psl_total': psl_total_eval, 'acc_psl': acc_psl, 'pslt_global': pslt_global,
                 'cw_acc_train': acc_train_cw, 'cw_acc_val': acc_val_cw, 'cw_avg_prob': cw_avg_prob.tolist(),
                 'local_threshold': local_threshold.tolist(), 'cw_psl_total_eval': cw_psl_total_eval.tolist(),
                 'cw_psl_correct_eval': cw_psl_correct_eval.tolist(), 'cw_psl_acc_eval': (cw_psl_correct_eval / cw_psl_total_eval).tolist(),
                 'cw_psl_total_accum': cw_psl_total_accum.tolist(), 'cw_psl_correct_accum': cw_psl_correct_accum.tolist(),
                 'cw_psl_acc_accum': (cw_psl_correct_accum / cw_psl_total_accum).tolist()})

            if acc_val > best_acc:
                best_epoch = epoch
                val_test_acc = acc_test
                val_test_f1 = f1_test

                best_acc = acc_val
                best_model_step = step
                early_stop_count = 0
                for netgroup in netgroups:
                    netgroup.save_model(output_dir_path, save_name, ema_mode=ema_mode)

            psl_total_eval, psl_correct_eval = 0, 0
            cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)

            step += 1
            for i, (x_lb, y_lb) in enumerate(batch_label):
                x_lb, y_lb = x_lb['x_w'], x_lb['label']
                sup_loss_nets = []

                for netgroup in netgroups:
                    outs_x_lb = netgroup.forward(x_lb, y_lb.to(device))
                    sup_loss_nets.append([ce_loss(outs_x_lb[j], y_lb.to(device)) for j in range(num_nets)])

                for netgroup, loss in zip(netgroups, sup_loss_nets):
                    netgroup.update(loss)
                    if ema_mode:
                        netgroup.update_ema()

            if load_mode == 'semi':
                for _ in range(ul_ratio):
                    batch_unlabel = [next(iter(loader)) for loader in train_unlabeled_loaders]

                    for i, (x_ulb_w, x_ulb_s) in enumerate(batch_unlabel):
                        x_ulb_w, x_ulb_s = x_ulb_w['x_w'], x_ulb_s['x_s']
                        outs_x_ulb_s_nets = [netgroup.forward(x_ulb_s) for netgroup in netgroups]
                        with torch.no_grad():
                            outs_x_ulb_w_nets = [netgroup.forward(x_ulb_w) for netgroup in netgroups]

                        pseudo_labels_nets = []
                        u_psl_masks_nets = []

                        for j, netgroup in enumerate(netgroups):
                            logits_x_ulb_w = outs_x_ulb_w_nets[j]
                            if labeling_mode == 'soft':
                                pseudo_labels_nets.append(torch.softmax(logits_x_ulb_w, dim=-1))
                            else:
                                max_idx = torch.argmax(logits_x_ulb_w, dim=-1)
                                pseudo_labels_nets.append(F.one_hot(max_idx, num_classes=n_classes).to(device))

                            probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
                            max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
                            if not adaptive_threshold:
                                pslt_global = psl_threshold_h
                                u_psl_masks_nets.append(max_probs >= pslt_global)
                            else:
                                pslt_global = psl_threshold_h
                                cw_avg_prob = cw_avg_prob * ema_momentum + torch.mean(probs_x_ulb_w, dim=0) * (1 - ema_momentum)
                                local_threshold = cw_avg_prob / torch.max(cw_avg_prob, dim=-1)[0]
                                u_psl_mask = max_probs.ge(pslt_global * local_threshold[max_idx])
                                u_psl_masks_nets.append(u_psl_mask)

                        total_unsup_loss_nets = []

                        for j, netgroup in enumerate(netgroups):
                            if cross_labeling:
                                pseudo_label = pseudo_labels_nets[(j + 1) % num_nets]
                                u_psl_mask = u_psl_masks_nets[(j + 1) % num_nets]
                            else:
                                pseudo_label = pseudo_labels_nets[j]
                                u_psl_mask = u_psl_masks_nets[j]

                            if weight_disagreement == 'True':
                                disagree_mask = torch.logical_xor(u_psl_masks_nets[j % num_nets], u_psl_masks_nets[(j + 1) % num_nets])
                                agree_mask = torch.logical_and(u_psl_masks_nets[j % num_nets], u_psl_masks_nets[(j + 1) % num_nets])
                                disagree_weight_masked = disagree_weight * disagree_mask + (1 - disagree_weight) * agree_mask
                            elif weight_disagreement == 'ablation_baseline':
                                disagree_weight = 0.5
                                disagree_mask = torch.logical_xor(u_psl_masks_nets[j % num_nets], u_psl_masks_nets[(j + 1) % num_nets])
                                agree_mask = torch.logical_and(u_psl_masks_nets[j % num_nets], u_psl_masks_nets[(j + 1) % num_nets])
                                disagree_weight_masked = disagree_weight * disagree_mask + (1 - disagree_weight) * agree_mask
                            else:
                                disagree_weight_masked = None

                            unsup_loss = consistency_loss(outs_x_ulb_s_nets[j], pseudo_label, loss_type='ce', mask=u_psl_mask, disagree_weight_masked=disagree_weight_masked)

                            total_unsup_loss = weight_u_loss * unsup_loss
                            total_unsup_loss_nets.append(total_unsup_loss)
                        for netgroup, loss in zip(netgroups, total_unsup_loss_nets):
                            netgroup.update(loss)
                            if ema_mode:
                                netgroup.update_ema()

                        batch_unlabel[i]['label'] = batch_unlabel[i]['label'].to(device)
                        gt_labels_u = batch_unlabel[i]['label'][u_psl_mask].to(device)
                        psl_total = torch.sum(u_psl_mask).item()

                        u_label_psl = pseudo_label[u_psl_mask]
                        u_label_psl_hard = torch.argmax(u_label_psl, dim=-1)
                        psl_correct = torch.sum(u_label_psl_hard == gt_labels_u).item()

                        psl_total_eval += psl_total
                        psl_correct_eval += psl_correct

                        cw_psl_total = torch.bincount(u_label_psl_hard, minlength=n_classes).to('cpu')
                        cw_psl_correct = torch.bincount(u_label_psl_hard[u_label_psl_hard == gt_labels_u], minlength=n_classes).to('cpu')

                        cw_psl_total_eval += cw_psl_total
                        cw_psl_correct_eval += cw_psl_correct

                        cw_psl_total_accum += cw_psl_total
                        cw_psl_correct_accum += cw_psl_correct

        pbar.write(f"Epoch {epoch + 1}/{max_epoch}, Train Acc: {acc_train:.4f}, "
            f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1(macro): {f1_test:.4f}, "
            f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}")
        pbar.update(1)
    pbar.close()

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - t0)))

    ##### Results Summary and Visualization
    # Load the saved best models and evaluate on the test set
    netgroup.load_model(output_dir_path, save_name, ema_mode=ema_mode)
    acc_test, f1_test, acc_test_cw, confmat_test = evaluation(test_loaders[0], final_eval=True)

    ## Quantitative Results
    pd.set_option('precision', 4)
    df_stats = pd.DataFrame(training_stats)
    df_stats = df_stats.set_index('step')
    training_stats_path = log_dir + 'training_statistics.csv'
    df_stats.to_csv(training_stats_path)
    print('Save training statistics in: ', training_stats_path)

    cur_time = time.strftime("%Y%m%d-%H%M%S")
    best_data = {'record_time': cur_time, 'best_step': best_model_step, 'test_acc': acc_test, 'test_f1': f1_test}

    print('\n\nBest_step: ', best_model_step, '\nBest_val_epoch: ', best_epoch + 1,
          '\nbest_val_acc: ', best_acc, '\nbest_val_test_acc: ', val_test_acc, '\nbest_val_test_f1: ', val_test_f1)

    best_data.update(params)
    best_df = pd.DataFrame([best_data])
    best_csv_path = log_dir_multiRun + 'summary.csv'
    if not os.path.exists(best_csv_path):
        best_df.to_csv(best_csv_path, mode='a', index=False, header=True)
    else:
        best_df.to_csv(best_csv_path, mode='a', index=False, header=False)
    best_df.to_csv(training_stats_path, mode='a', index=False, header=True)
    print('Save best record in: ', best_csv_path, end='')

    ## Visualization - Plot Training Curves
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_stats_1 = df_stats
    plot_types = ['f1', 'acc', 'psl', 'pslt']

    for plot_type in plot_types:
        plt.figure()
        sns.set(style='darkgrid')
        sns.set(font_scale=1.5)

        for idx, key in enumerate(df_stats.keys().tolist()):
            if key.split('_')[0] == plot_type:
                plt.plot(df_stats_1[key], '--', label=key)

        plt.xlabel("iteration")
        plt.ylabel("performance")
        plt.legend()
        plt.savefig(log_dir + plot_type + '.png', bbox_inches='tight')

    df_cm = pd.DataFrame(confmat_test, index=[i for i in range(n_classes)],
                         columns=[i for i in range(n_classes)], dtype=int)
    df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(20, 14))
    sns.heatmap(df_cm_norm, annot=True, annot_kws={"size": 10}, fmt='.2f', cmap='Reds')
    plt.savefig(log_dir + 'confmat_norm.png', bbox_inches='tight')
    return best_data
