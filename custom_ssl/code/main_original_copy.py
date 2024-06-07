import os
import random
import sys
import numpy as np
import pandas as pd
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
    n_labeled_per_class = 10 if 'n_labeled_per_class' not in params else params['n_labeled_per_class']
    bs = 8 if 'bs' not in params else params['bs']  # original: 32
    ul_ratio = 10 if 'ul_ratio' not in params else params['ul_ratio']
    lr = 2e-5 if 'lr' not in params else params['lr']  # original: 1e-4, 2e-5
    lr_linear = 1e-3 if 'lr_linear' not in params else params['lr_linear']  # original: 1e-3

    # - semi-supervised
    weight_u_loss = 0.1 if 'weight_u_loss' not in params else params['weight_u_loss']
    load_mode = 'semi' if 'load_mode' not in params else params['load_mode']  # semi, sup_baseline

    # - pseudo-labeling
    psl_threshold_h = 0.98 if 'psl_threshold_h' not in params else params['psl_threshold_h']  # original: 0.75
    labeling_mode = 'hard' if 'labeling_mode' not in params else params['labeling_mode']  # hard, soft
    adaptive_threshold = False if 'adaptive_threshold' not in params else params['adaptive_threshold']  # True, False

    # - ensemble
    num_nets = 2 if 'num_nets' not in params else params['num_nets']
    cross_labeling = False if 'cross_labeling' not in params else params['cross_labeling']  # True, False

    # - weight disagreement
    weight_disagreement = False if 'weight_disagreement' not in params else params['weight_disagreement']  # True, False
    disagree_weight = 1 if 'disagree_weight' not in params else params['disagree_weight']

    # - ema
    ema_mode = False if 'ema_mode' not in params else params['ema_mode']
    ema_momentum = 0.9 if 'ema_momentum' not in params else params['ema_momentum']  # original: 0.99

    # - others
    seed = params['seed']
    device_idx = 1 if 'device_idx' not in params else params['device_idx']
    val_interval = 25 if 'val_interval' not in params else params['val_interval']  # 20, 25
    early_stop_tolerance = 10 if 'early_stop_tolerance' not in params else params['early_stop_tolerance']  # 5, 6, 10
    max_epoch = 10000 if 'max_epoch' not in params else params['max_epoch']  # 100, 200
    max_step = 100000 if 'max_step' not in params else params['max_step']  # 100000, 200000

    # 추가
    net_archs = params['net_arch']
    tokens = ["microsoft/codebert-base", "roberta-base"] if 'token' not in params else params['token']
    save_name = 'pls_save_name' if 'save_name' not in params else params['save_name']

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
    from utils.dataloader_original_copy import get_dataloader
    print("\n**line 107 모델 => ", net_archs)
    print('**tokenizer type = ', tokens, '\n')

    # Load data for each model
    train_labeled_loader_1, train_unlabeled_loader_1, dev_loader_1, test_loader_1, n_classes = get_dataloader(data_path, n_labeled_per_class, bs, load_mode, tokens[0])
    print('')
    train_labeled_loader_2, train_unlabeled_loader_2, dev_loader_2, test_loader_2, _ = get_dataloader(data_path, n_labeled_per_class, bs, load_mode, tokens[1])
    
    print('\nn_classes: ', n_classes)

    ##### Model & Optimizer & Learning Rate Scheduler #####
    from models.netgroup_original_copy import NetGroup

    # Initialize model & optimizer & lr_scheduler
    netgroup1 = NetGroup(net_archs[0], num_nets, n_classes, device, lr[0], lr_linear)
    netgroup2 = NetGroup(net_archs[1], num_nets, n_classes, device, lr[1], lr_linear)

    # Initialize EMA
    netgroup1.train()
    netgroup2.train()
    if ema_mode:
        netgroup1.init_ema(ema_momentum)
        netgroup2.init_ema(ema_momentum)

    ##### Training & Evaluation #####
    ## Set or import criterions & helper functions
    from utils.helper import format_time
    from criterions.criterions import ce_loss, consistency_loss

    ## Evaluation
    from sklearn.metrics import f1_score, accuracy_score
    from torchmetrics import Accuracy
    from torchmetrics.classification import MulticlassConfusionMatrix

    accuracy_classwise = Accuracy(num_classes=n_classes, average='none')
    confusion_matrix = MulticlassConfusionMatrix(num_classes=n_classes)

    @torch.no_grad()  # no need to track gradients in validation
    def evaluation(loader_1, loader_2, final_eval=False):
        """Evaluation"""
        # Put the models in evaluation mode
        netgroup1.eval()
        netgroup2.eval()
        if ema_mode:  # use ema model for evaluation
            netgroup1.eval_ema()
            netgroup2.eval_ema()

        # Tracking variables
        preds_all = []
        target_all = []

        # Evaluate data for one epoch
        for batch_1, batch_2 in zip(loader_1, loader_2):
            b_labels = batch_1['label'].to(device)

            # Forward pass for both networks
            outs1 = netgroup1.forward(batch_1['x'], b_labels)
            outs2 = netgroup2.forward(batch_2['x'], b_labels)

            # Assuming outs1 and outs2 are lists of logits, we take the first element (logits) for softmax
            logits1 = outs1[0] if isinstance(outs1, list) else outs1
            logits2 = outs2[0] if isinstance(outs2, list) else outs2

            # Take average probs from both nets
            probs1 = torch.softmax(logits1, dim=-1)
            probs2 = torch.softmax(logits2, dim=-1)
            probs = torch.mean(torch.stack([probs1, probs2]), dim=0)

            # Move preds and labels to CPU
            preds = torch.argmax(probs, dim=1)
            target = b_labels

            # For calculating classwise acc
            preds_all.append(preds)
            target_all.append(target)

        # Calculate acc and macro-F1
        preds_all = torch.cat(preds_all).detach().cpu()
        target_all = torch.cat(target_all).detach().cpu()
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
    import time
    from utils.helper import freematch_fairness_loss
    from tqdm import tqdm

    # Initialize variables
    t0 = time.time()  # Measure how long the training takes.
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
    cw_avg_prob = (torch.ones(n_classes) / n_classes).to(device)  # estimate learning status of each class
    local_threshold = torch.zeros(n_classes, dtype=int)

    # Training
    netgroup1.train()
    netgroup2.train()

    pbar = tqdm(total=max_epoch, desc="Training", position=0, leave=True)

    for epoch in range(max_epoch):
        pse_table = []
        if data_path.split('/')[2] == 'corcode_extended_data':
            pse_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        else:
            pse_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

        if step > max_step:
            print("조기종료 step > max step =>", step > max_step)
            break

        for batch_label_1, batch_label_2 in zip(train_labeled_loader_1, train_labeled_loader_2):
            acc_test, f1_test, acc_test_cw = evaluation(test_loader_1, test_loader_2)
            acc_val, f1_val, acc_val_cw = evaluation(dev_loader_1, dev_loader_2)
            acc_train, f1_train, acc_train_cw = evaluation(train_labeled_loader_1, train_labeled_loader_2)

            if ema_mode:
                netgroup1.train_ema()
                netgroup2.train_ema()
            netgroup1.train()
            netgroup2.train()

            acc_psl = (psl_correct_eval / psl_total_eval) if psl_total_eval > 0 else None

            training_stats.append({
                'step': step,
                'acc_val': acc_val,
                'f1_val': f1_val,
                'acc_train': acc_train,
                'f1_train': f1_train,
                'psl_correct': psl_correct_eval,
                'psl_total': psl_total_eval,
                'acc_psl': acc_psl,
                'pslt_global': pslt_global,
                'cw_acc_train': acc_train_cw,
                'cw_acc_val': acc_val_cw,
                'cw_avg_prob': cw_avg_prob.tolist(),
                'local_threshold': local_threshold.tolist(),
                'cw_psl_total_eval': cw_psl_total_eval.tolist(),
                'cw_psl_correct_eval': cw_psl_correct_eval.tolist(),
                'cw_psl_acc_eval': (cw_psl_correct_eval / cw_psl_total_eval).tolist(),
                'cw_psl_total_accum': cw_psl_total_accum.tolist(),
                'cw_psl_correct_accum': cw_psl_correct_accum.tolist(),
                'cw_psl_acc_accum': (cw_psl_correct_accum / cw_psl_total_accum).tolist(),
            })

            if acc_val > best_acc:
                best_epoch = epoch
                val_test_acc = acc_test
                val_test_f1 = f1_test

                best_acc = acc_val
                best_model_step = step
                early_stop_count = 0
                netgroup1.save_model(output_dir_path, save_name, ema_mode=ema_mode)
                netgroup2.save_model(output_dir_path, save_name + '_2', ema_mode=ema_mode)

            psl_total_eval, psl_correct_eval = 0, 0
            cw_psl_total_eval, cw_psl_correct_eval = torch.zeros(n_classes, dtype=int), torch.zeros(n_classes, dtype=int)

            step += 1
            x_lb_1, y_lb_1 = batch_label_1['x_w'], batch_label_1['label']
            x_lb_2, y_lb_2 = batch_label_2['x_w'], batch_label_2['label']
            outs_x_lb_1 = netgroup1.forward(x_lb_1, y_lb_1.to(device))
            outs_x_lb_2 = netgroup2.forward(x_lb_2, y_lb_2.to(device))
            sup_loss_nets_1 = ce_loss(outs_x_lb_1[0], y_lb_1.to(device))
            sup_loss_nets_2 = ce_loss(outs_x_lb_2[0], y_lb_2.to(device))

            netgroup1.update([sup_loss_nets_1])
            netgroup2.update([sup_loss_nets_2])
            if ema_mode:
                netgroup1.update_ema()
                netgroup2.update_ema()

            if load_mode == 'semi':
                for _ in range(ul_ratio):
                    try:
                        batch_unlabel_1 = next(data_iter_unl_1)
                    except:
                        data_iter_unl_1 = iter(train_unlabeled_loader_1)
                        batch_unlabel_1 = next(data_iter_unl_1)
                    
                    try:
                        batch_unlabel_2 = next(data_iter_unl_2)
                    except:
                        data_iter_unl_2 = iter(train_unlabeled_loader_2)
                        batch_unlabel_2 = next(data_iter_unl_2)

                    x_ulb_w_1, x_ulb_s_1 = batch_unlabel_1['x_w'], batch_unlabel_1['x_s']
                    x_ulb_w_2, x_ulb_s_2 = batch_unlabel_2['x_w'], batch_unlabel_2['x_s']

                    outs_x_ulb_s_1 = netgroup1.forward(x_ulb_s_1)
                    outs_x_ulb_s_2 = netgroup2.forward(x_ulb_s_2)
                    with torch.no_grad():
                        outs_x_ulb_w_1 = netgroup1.forward(x_ulb_w_1)
                        outs_x_ulb_w_2 = netgroup2.forward(x_ulb_w_2)

                    pseudo_labels_nets = []
                    u_psl_masks_nets = []

                    logits_x_ulb_w_1 = outs_x_ulb_w_1[0]
                    pseudo_labels_1 = torch.softmax(logits_x_ulb_w_1, dim=-1) if labeling_mode == 'soft' else F.one_hot(torch.argmax(logits_x_ulb_w_1, dim=-1), num_classes=n_classes).to(device)

                    logits_x_ulb_w_2 = outs_x_ulb_w_2[0]
                    pseudo_labels_2 = torch.softmax(logits_x_ulb_w_2, dim=-1) if labeling_mode == 'soft' else F.one_hot(torch.argmax(logits_x_ulb_w_2, dim=-1), num_classes=n_classes).to(device)

                    probs_x_ulb_w_1 = torch.softmax(logits_x_ulb_w_1, dim=-1)
                    max_probs_1, max_idx_1 = torch.max(probs_x_ulb_w_1, dim=-1)
                    u_psl_mask_1 = max_probs_1 >= pslt_global

                    probs_x_ulb_w_2 = torch.softmax(logits_x_ulb_w_2, dim=-1)
                    max_probs_2, max_idx_2 = torch.max(probs_x_ulb_w_2, dim=-1)
                    u_psl_mask_2 = max_probs_2 >= pslt_global

                    pseudo_labels_nets.append(pseudo_labels_1)
                    u_psl_masks_nets.append(u_psl_mask_1)
                    pseudo_labels_nets.append(pseudo_labels_2)
                    u_psl_masks_nets.append(u_psl_mask_2)

                    total_unsup_loss_nets = []
                    for i in range(num_nets):
                        pseudo_label = pseudo_labels_nets[(i + 1) % 2]
                        u_psl_mask = u_psl_masks_nets[(i + 1) % 2]

                        unsup_loss = consistency_loss(netgroup1.forward(x_ulb_s_1)[i], pseudo_label, loss_type='ce', mask=u_psl_mask) if i == 0 else consistency_loss(netgroup2.forward(x_ulb_s_2)[i], pseudo_label, loss_type='ce', mask=u_psl_mask)
                        total_unsup_loss = weight_u_loss * unsup_loss
                        total_unsup_loss_nets.append(total_unsup_loss)

                    import pdb
                    pdb.set_trace()
                    netgroup1.update([total_unsup_loss_nets[0]])
                    netgroup2.update([total_unsup_loss_nets[1]])
                    if ema_mode:
                        netgroup1.update_ema()
                        netgroup2.update_ema()

                    batch_unlabel_1['label'] = batch_unlabel_1['label'].to(device)
                    batch_unlabel_2['label'] = batch_unlabel_2['label'].to(device)
                    gt_labels_u_1 = batch_unlabel_1['label'][u_psl_mask_1].to(device)
                    gt_labels_u_2 = batch_unlabel_2['label'][u_psl_mask_2].to(device)
                    psl_total_1 = torch.sum(u_psl_mask_1).item()
                    psl_total_2 = torch.sum(u_psl_mask_2).item()

                    u_label_psl_1 = pseudo_labels_1[u_psl_mask_1]
                    u_label_psl_hard_1 = torch.argmax(u_label_psl_1, dim=-1)
                    psl_correct_1 = torch.sum(u_label_psl_hard_1 == gt_labels_u_1).item()

                    u_label_psl_2 = pseudo_labels_2[u_psl_mask_2]
                    u_label_psl_hard_2 = torch.argmax(u_label_psl_2, dim=-1)
                    psl_correct_2 = torch.sum(u_label_psl_hard_2 == gt_labels_u_2).item()

                    psl_total_eval += psl_total_1 + psl_total_2
                    psl_correct_eval += psl_correct_1 + psl_correct_2

                    cw_psl_total_1 = torch.bincount(u_label_psl_hard_1, minlength=n_classes).to('cpu')
                    cw_psl_correct_1 = torch.bincount(u_label_psl_hard_1[u_label_psl_hard_1 == gt_labels_u_1], minlength=n_classes).to('cpu')

                    cw_psl_total_2 = torch.bincount(u_label_psl_hard_2, minlength=n_classes).to('cpu')
                    cw_psl_correct_2 = torch.bincount(u_label_psl_hard_2[u_label_psl_hard_2 == gt_labels_u_2], minlength=n_classes).to('cpu')

                    cw_psl_total_eval += cw_psl_total_1 + cw_psl_total_2
                    cw_psl_correct_eval += cw_psl_correct_1 + cw_psl_correct_2

                    cw_psl_total_accum += cw_psl_total_1 + cw_psl_total_2
                    cw_psl_correct_accum += cw_psl_correct_1 + cw_psl_correct_2

        pbar.write(f"Epoch {epoch + 1}/{max_epoch}, Train Acc: {acc_train:.4f}, "
                   f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1(macro): {f1_test:.4f}, "
                   f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}")
        pbar.update(1)
    pbar.close()

    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - t0)))

    ##### Results Summary and Visualization
    netgroup1.load_model(output_dir_path, save_name, ema_mode=ema_mode)
    netgroup2.load_model(output_dir_path, save_name + '_2', ema_mode=ema_mode)
    acc_test, f1_test, acc_test_cw, confmat_test = evaluation(test_loader_1, test_loader_2, final_eval=True)

    ## Quantatitive Results
    pd.set_option('precision', 4)
    df_stats = pd.DataFrame(training_stats)
    df_stats = df_stats.set_index('step')
    training_stats_path = log_dir + 'training_statistics.csv'
    df_stats.to_csv(training_stats_path)
    print('Save training statistics in: ', training_stats_path)

    cur_time = time.strftime("%Y%m%d-%H%M%S")
    best_data = {'record_time': cur_time,
                 'best_step': best_model_step, 'test_acc': acc_test, 'test_f1': f1_test,
                 }

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
        plt.ylabel("peformance")
        plt.legend()
        plt.savefig(log_dir + plot_type + '.png', bbox_inches='tight')

    df_cm = pd.DataFrame(confmat_test, index=[i for i in range(n_classes)],
                         columns=[i for i in range(n_classes)], dtype=int)
    df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0)
    plt.figure(figsize=(20, 14))
    sns.heatmap(df_cm_norm, annot=True, annot_kws={"size": 10}, fmt='.2f', cmap='Reds')
    plt.savefig(log_dir + 'confmat_norm.png', bbox_inches='tight')

    return best_data


##### multiRun ######
def multiRun(experiment_home=None, num_runs=3, unit_test_mode=False, **params):
    import statistics
    import pandas as pd
    import os
    import random
    import time

    ## Create folder structure
    seeds_list = params['seeds_list']
    cur_time = time.strftime("%Y%m%d-%H%M%S")
    if experiment_home is None:
        experiment_home = root + '/experiment/temp/'
    log_root = experiment_home + '/log/'
    global log_dir_multiRun
    log_dir_multiRun = log_root + f"{params['model_name']}/joint_match" + '/'

    if not os.path.exists(experiment_home):
        os.makedirs(experiment_home)
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(log_dir_multiRun):
        os.makedirs(log_dir_multiRun)

    for i in range(num_runs):
        log_dir = log_dir_multiRun + '/' + params['save_name']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    output_dir_experiment = experiment_home + '/output/'
    if not os.path.exists(output_dir_experiment):
        os.makedirs(output_dir_experiment)

    results = []
    test_accs = []
    test_f1s = []
    for i in range(num_runs):
        if not unit_test_mode:
            log_dir = log_dir_multiRun + '/' + params['save_name'] + '/'
            result = oneRun(log_dir, output_dir_experiment, **params, seed=seeds_list[0])
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
                 'Mean_std_acc': '%.2f ± %.2f' % (100 * mean_acc, 100 * st_dev_acc),
                 'Mean_std_f1': '%.2f ± %.2f' % (100 * mean_f1, 100 * st_dev_f1)}
    else:
        final = {'record_time': cur_time,
                 'Mean_std_acc': '%.2f' % (100 * mean_acc),
                 'Mean_std_f1': '%.2f' % (100 * mean_f1)}

    final.update(params)
    final.update({'seeds_list': seeds_list})
    df = pd.DataFrame([final])
    csv_path = log_root + 'summary_avgrun.csv'
    df.to_csv(csv_path, mode='a', index=False, header=True)
    print('\nSave best record in: ', csv_path)
