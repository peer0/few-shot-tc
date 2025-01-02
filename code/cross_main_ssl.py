import os
import time
import argparse
import json
import random
import statistics
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from models.netgroup import NetGroup
from utils.helper import format_time
from utils.cross_dataloader import get_dataloader_v3
from utils.cross_dataloader import get_dataloader_v4
from criterions.criterions import ce_loss, consistency_loss
from utils.helper import freematch_fairness_loss
from utils.cross_dataloader import MyCollator_SSL, BalancedBatchSampler
from symbolic.symbolic import process_code


def pseudo_labeling(netgroup,num_nets,train_unlabeled_loader, psl_threshold_h, device, pbar, language):
    psl_correct = 0
    psl_total = 0
    idx = 0
    pseudo_label_temp = []
    if language == 'corcod':
        pseudo_labels = {1:[], 2:[], 3:[], 4:[], 5:[]}
    else:
        pseudo_labels = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
    train_dataset_u = train_unlabeled_loader.dataset
    for batch_unlabel in train_unlabeled_loader:
        # in the loader, the range of labels are from 0 to 6
        # in the dataset, the range of labels are from 1 to 7
        origin_sent = train_dataset_u.sents[idx]
        origin_sent_aug1 = train_dataset_u.sents_aug1[idx]
        origin_sent_aug2 = train_dataset_u.sents_aug2[idx]
        answer_label = train_dataset_u.labels[idx]-1
        x_ulb_s = batch_unlabel['x']
        with torch.no_grad():
            outs_x_ulb_w_nets = netgroup.forward(x_ulb_s)
        for i in range(num_nets):
            logits_x_ulb_w = outs_x_ulb_w_nets[i]
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
            assert batch_unlabel['label'].item() == answer_label
            #if max_probs > 0.6:
            #    pbar.write(f"Model confidence: {max_probs}, Predict label: {max_idx}, Reference label: {batch_unlabel['label'].item()}")
            for target_idx, target_probs in zip(max_idx, max_probs):
                target_idx = int(target_idx)
                if target_probs >= psl_threshold_h:
                    if target_idx == batch_unlabel['label'].item():
                        psl_correct += 1
                    # Update dataset with pseudo-label
                    pseudo_label_temp.append({"codes":{"original":origin_sent, "backtrans":origin_sent_aug1, "forwhile": origin_sent_aug2}, "label":int(target_idx + 1)})  # Add pseudo-labeled data to the labeled dataset
                elif target_probs < psl_threshold_h:
                    symbolic_prediction = process_code(origin_sent, language)
                    if symbolic_prediction > 0:
                        if symbolic_prediction-1 == batch_unlabel['label'].item():
                            psl_correct += 1
                        pseudo_label_temp.append({"codes":{"original":origin_sent, "backtrans":origin_sent_aug1, "forwhile": origin_sent_aug2}, "label":symbolic_prediction})  # Add pseudo-labeled data to the labeled dataset
        idx+=1

    for i in pseudo_label_temp:
        if i['label'] not in pseudo_labels: continue
        pseudo_labels[i['label']].append(i['codes'])


    return pseudo_labels, len(pseudo_label_temp), psl_correct

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

def train_one_epoch(netgroup, train_labeled_loader, train_unlabeled_loader, n_classes, device, params):
    netgroup.train()
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    disagree_weight = 0.5
    num_nets = params['num_nets'] + 1
    for batch_label in train_labeled_loader:
        x_lb, y_lb = batch_label['x'], batch_label['label'].to(device)
        x_lb_w, x_lb_s = batch_label['x_w'], batch_label['x_s']
        outs_x_lb = netgroup.forward(x_lb, y_lb)
        sup_loss_nets = [ce_loss(outs_x_lb[i], y_lb) for i in range(num_nets)]
        netgroup.update(sup_loss_nets)
        for sup_loss_net in sup_loss_nets:
            total_sup_loss += sup_loss_net.item()


        pseudo_labels_nets = []
        u_psl_masks_nets = []

        x_lb_w, x_lb_s = batch_label['x_w'], batch_label['x_s']
        outs_x_ulb_w_nets = netgroup.forward(x_lb_w, y_lb)
        with torch.no_grad(): # stop gradient for weak augmentation brach
            outs_x_ulb_s_nets = netgroup.forward(x_lb_s, y_lb)

        for i in range(num_nets):
            ## generate pseudo labels
            logits_x_ulb_s = outs_x_ulb_s_nets[i]
            max_idx = torch.argmax(logits_x_ulb_s, dim=-1)
            pseudo_labels_nets.append(torch.softmax(logits_x_ulb_s, dim=-1))
            #pseudo_labels_nets.append(F.one_hot(max_idx, num_classes=n_classes).to(device))

            ## compute mask for pseudo labels
            probs_x_ulb_s = torch.softmax(logits_x_ulb_s, dim=-1)
            max_probs, max_idx = torch.max(probs_x_ulb_s, dim=-1)
            u_psl_masks_nets.append(max_probs >= params['psl_threshold_h'])

        ## Compute loss for unlabeled data for all nets
        total_unsup_loss_nets = []
        for i in range(num_nets):
            pseudo_label = pseudo_labels_nets[(i+1)%num_nets]
            u_psl_mask = u_psl_masks_nets[(i+1)%num_nets]

            # obtain the mask for disagreement and agreement across nets, note that they are derived from confident pseudo-label mask
            # disagree_weight, agree_weight can be a specified scalar or a tensor calculated based on disagreement score
            disagree_mask = torch.logical_xor(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
            agree_mask = torch.logical_and(u_psl_masks_nets[(i)%num_nets], u_psl_masks_nets[(i+1)%num_nets])
            disagree_weight_masked = disagree_weight * disagree_mask + (1-disagree_weight) * agree_mask

            # compute loss for unlabeled data
            unsup_loss = consistency_loss(outs_x_ulb_w_nets[i], pseudo_label, loss_type='ce', mask=u_psl_mask, disagree_weight_masked=disagree_weight_masked)    # loss_type: 'ce' or 'mse'

            # compute total loss for unlabeled data
            #total_unsup_loss = weight_u_loss * unsup_loss
            total_unsup_loss_nets.append(unsup_loss)
            total_unsup_loss += unsup_loss.item()

        # update netgorup from loss of unlabeled data
        netgroup.update(total_unsup_loss_nets)


    return total_sup_loss / len(train_labeled_loader), total_unsup_loss / len(train_unlabeled_loader)

def train(output_dir_path, seed, params):
    train_sup_losses = []
    train_unsup_losses = []
    val_losses = []
    training_stats = []
    if params['dataset'] =='python':
        language = 'python'
    elif params['dataset'] =='java':
        language = 'java'
    elif params['dataset'] =='corcod':
        language = 'corcod'
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params['aug'] == 'natural':
        train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader_v3(
            '../data/' + params['dataset'],params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
            params['net_arch'])
    elif params['aug'] == 'artificial':
        train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader_v4(
            '../data/' + params['dataset'],params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
            params['net_arch'])


    print(n_classes)
    print(f"Complexity Class Number: {n_classes}")
    print(f"Initial Train Data Number: {len(train_dataset_l)}")
    print(f"Initial Unlabeled Train Data Number: {len(train_dataset_u)}")
    print(f"Valid Data Number: {len(dev_loader)}")
    print(f"Test Data Number: {len(test_loader)}")

    # Initialize model
    netgroup = NetGroup(params['net_arch'], params['num_nets']+1, n_classes, device, params['lr'])
    netgroup.to(device)
    tokenizer = AutoTokenizer.from_pretrained(params['net_arch'])
    best_train_dataset_l = train_dataset_l

    # Load data
    train_unlabeled_loader = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False,
                                        collate_fn=MyCollator_SSL(tokenizer))

    
    best_checkpoint_acc_val = 0.0
    best_checkpoint_val_loss = 1000.0
    best_checkpoint_acc_test = 0.0
    best_checkpoint_epoch = 0.0
    min_length = 0
    labels_with_min_length = 0
    best_checkpoint_f1_macro_test = 0.0
    best_conf_matrix = None

    pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
    for epoch in range(params["max_epoch"]):
        epoch_train_num = len(train_dataset_l)
        # Update train labeled loader with new pseudo-labeled data
        # train_labeled_loader.dataset.update_data(train_dataset_u)
        train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
        train_sup_loss, train_unsup_loss = train_one_epoch(netgroup, train_labeled_loader, train_unlabeled_loader, n_classes, device, params)
        train_sup_losses.append(train_sup_loss)
        train_unsup_losses.append(train_unsup_loss)
        val_loss = calculate_loss(netgroup, dev_loader, device)
        val_losses.append(val_loss)

        # Evaluate
        acc_train, _, _ = evaluate(netgroup, train_labeled_loader, device)
        acc_val, _, _ = evaluate(netgroup, dev_loader, device)
        acc_test, f1_macro_test, conf_matrix = evaluate(netgroup, test_loader, device)
        pseudo_labels, psl_total, psl_correct = pseudo_labeling(netgroup, params['num_nets']+1, train_unlabeled_loader, params['psl_threshold_h'], device, pbar, language)

        training_stats.append(
            {   'cross': 'cross',
                'epoch': epoch, #배치수
                'acc_train': acc_train,#train의 acc
                'acc_val': acc_val,#valid의 acc
                'acc_test': acc_test,#test의 acc
                'f1_test': f1_macro_test, #test의 f1 macro
                'acc_psl': 2 * psl_correct / psl_total, # 
                'psl_correct': psl_correct, # 
                'psl_total': psl_total,
            })

        # decide whether the pseudo_label list is empty
        if sum(1 for codes in pseudo_labels.values() if len(codes) >= 1):
        
            min_length = min(len(codes) for codes in pseudo_labels.values() if codes)

            # Calculate how many labels can provide at least 'min_length' amount of codes
            labels_with_min_length = sum(1 for codes in pseudo_labels.values() if len(codes) >= min_length)

            # Randomly select 'min_length' codes from each label's list and add to the train dataset
            for label, codes in pseudo_labels.items():
                if len(codes) > 0:  # Ensure the list is not empty
                    selected_codes = random.sample(codes, min_length)
                    for code in selected_codes:
                        train_dataset_l.add_data(code['original'], code['backtrans'], code['forwhile'], label)
        

        if params["checkpoint"] == 'loss':
            if val_loss < best_checkpoint_val_loss:
                best_checkpoint_acc_test = acc_test
                best_checkpoint_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                best_conf_matrix = conf_matrix
                netgroup.save_model(output_dir_path, "cross-{}.{}".format(params["lr"],params["acc_save_name"]))
        elif params["checkpoint"] == 'acc':
            if acc_test > best_checkpoint_acc_test:
                best_checkpoint_acc_test = acc_test
                best_checkpoint_f1_macro_test = f1_macro_test
                best_checkpoint_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                best_conf_matrix = conf_matrix
                netgroup.save_model(output_dir_path, "cross-{}.{}".format(params["lr"],params["acc_save_name"]))
        
        pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Sup Loss: {train_sup_loss:.4f}, Train Unsup Loss: {train_unsup_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                   f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1 Macro: {f1_macro_test:.4f}, "
                   f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}, "
                   f"Train Data Number: {epoch_train_num} + {min_length} x {labels_with_min_length} = {len(train_dataset_l)}")
        pbar.update(1)

    pd.set_option('precision', 4)
    df_stats= pd.DataFrame(training_stats)
    df_stats = df_stats.set_index('epoch')
    training_stats_path = os.path.join(output_dir_path,'cross-training_statistics.csv')
    df_stats.to_csv(training_stats_path)
    
    labels = [
        r'$O(1)$',  # constant 0
        r'$O(\log N)$',  # logn 1
        r'$O(N)$',  # linear 2
        r'$O(N \log N)$',  # nlogn 3
        r'$O(N^2)$',  # quadratic 4
        r'$O(N^3)$',  # cubic 5
        r'$O(2^N)$',  # exponential 6
    ]

    plot_types = ['f1', 'acc', 'psl']

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
                plt.plot(df_stats[key], '--', label=key)
        # Label the plot.
        plt.xlabel("iteration")
        plt.ylabel("peformance")
        plt.legend()
        plt.savefig(os.path.join(output_dir_path, "cross-{}.{}.png".format(params['lr'],plot_type)), bbox_inches='tight')
        plt.close()  # Closes the plot

    # loss 값의 변화를 그래프로 시각화합니다.
    plt.figure(figsize=(20,14))
    #epochs = range(1, max_epoch + 1)  # epoch 수
    plt.plot(train_sup_losses, label='Train Lab. Loss')
    plt.plot(train_unsup_losses, label='Train Unlab. Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir_path,'cross-{}.loss_plot.png'.format(params['lr'])), bbox_inches='tight')
    plt.close()  # Closes the plot

    # Plotting the confusion matrix
    plt.figure(figsize=(11,9))
    ax = sns.heatmap(best_conf_matrix, annot=True, fmt='.2f', cmap='OrRd',
            xticklabels=labels, yticklabels=labels, annot_kws={"size":16})

    ax.tick_params(axis='x', labelsize=16)  # Adjust x-axis label size
    ax.tick_params(axis='y', labelsize=16)  # Adjust y-axis label size

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # Adjust colorbar label size

    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    plt.savefig(os.path.join(output_dir_path, "cross-{}.confmat.png".format(params['lr'])), bbox_inches='tight')  # Saves the plot as a PNG file
    plt.close()  # Closes the plot

    pbar.write(f"(Valid {params['checkpoint']}) Best Epoch: {best_checkpoint_epoch}, Best Test Accuracy: {best_checkpoint_acc_test}, \
                Best Test F1 Macro: {best_checkpoint_f1_macro_test:.4f}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
    pbar.close()
    torch.cuda.empty_cache()
    return best_checkpoint_epoch, best_checkpoint_acc_test, best_checkpoint_f1_macro_test


def evaluate(netgroup, loader, device):
    netgroup.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            b_labels = batch['label'].to(device)
            outs = netgroup.forward(batch['x'], b_labels)
            preds = torch.argmax(torch.mean(torch.softmax(torch.stack(outs), dim=2), dim=0), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
    encoded_all_preds = []
    encoded_all_labels = []

    labels = [
        r'$O(1)$',  # constant 0
        r'$O(\log N)$',  # logn 1
        r'$O(N)$',  # linear 2
        r'$O(N \log N)$',  # nlogn 3
        r'$O(N^2)$',  # quadratic 4
        r'$O(N^3)$',  # cubic 5
        r'$O(2^N)$',  # exponential 6
    ]
    class_complexity_dict = {0: r'$O(1)$', 1: r'$O(\log N)$', 2: r'$O(N)$', 3: r'$O(N \log N)$', 4: r'$O(N^2)$', 5: r'$O(N^3)$', 6: r'$O(2^N)$'}
    for l,p in zip(all_labels, all_preds):
        encoded_all_labels.append(class_complexity_dict[l])
        encoded_all_preds.append(class_complexity_dict[p])

    # Generating the confusion matrix
    conf_matrix = confusion_matrix(encoded_all_labels, encoded_all_preds, labels=labels, normalize='true')

    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro'), conf_matrix

def main(config_file='config.json', **kwargs):
    # Load parameters from config file
    params = load_config(config_file)
    best_epochs = []
    best_accs = []
    best_f1s_macro = []

    # Override parameters from config file with command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    output_dir_path = './experiment/{}_{}_{}_{}/'.format(params['dataset'], params['model_name'], params['aug'], params['n_labeled_per_class'])
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Use the merged parameters for further processing
    print("Merged parameters:", params)

    # Train
    for i in range(3):
        seed = params['seed']+i
        output_seed_path = './experiment/{}_{}_{}_{}/{}'.format(params['dataset'], params['model_name'], params['aug'], params['n_labeled_per_class'],seed)
        if not os.path.exists(output_seed_path):
            os.makedirs(output_seed_path)
        best_epoch, best_acc, best_f1_macro = train(output_seed_path, seed, params)
        best_epochs.append(best_epoch)
        best_accs.append(best_acc)
        best_f1s_macro.append(best_f1_macro)

    st_dev_acc = round(statistics.pstdev(best_accs), 4)
    mean_acc = round(statistics.mean(best_accs), 4)
    st_dev_f1_macro = round(statistics.pstdev(best_f1s_macro), 4)
    mean_f1_macro = round(statistics.mean(best_f1s_macro), 4)

    final = {
        'dataset': params['dataset'],
        'shots': params['n_labeled_per_class'],
        'model name': params['model_name'],
        'augmentation': params['aug'],
        'Mean_std_acc': '%.2f \\pm %.2f' % (100*mean_acc, 100*st_dev_acc),
        'Mean_std_f1_macro': '%.2f \\pm %.2f' % (100*mean_f1_macro, 100*st_dev_f1_macro),
    }

    # Save best model info
    df = pd.DataFrame([final])
    csv_path = output_dir_path + 'cross-summary_avgrun.csv'
    df.to_csv(csv_path, mode='a', index=False, header=True)
    print('\nSave best record in: ', csv_path)
    print("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description here')

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--n_labeled_per_class', type=int, required=True, help='Number of labeled samples per class')
    parser.add_argument('--psl_threshold_h', type=float, required=True, help='Threshold for pseudo-labeling')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--checkpoint', type=str, help='retrieve the best valid loss checkpoint or valid acc checkpoint')
    parser.add_argument('--aug', type=str, default='natural', help='augment setting: natural, artificial, none (no augment)')

    args = parser.parse_args()
    return vars(args)

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    args = parse_args()
    config_file = args['config']
    main(config_file, **args)
