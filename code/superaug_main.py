import os
import time
import argparse
import json
import random
import statistics
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

from models.netgroup import NetGroup
from utils.helper import format_time
from utils.supaug_dataloader import get_dataloader_single
from utils.supaug_dataloader import get_dataloader_all
from utils.supaug_dataloader import get_dataloader_sup
from criterions.criterions import ce_loss, consistency_loss
from utils.helper import freematch_fairness_loss
from utils.supaug_dataloader import MyCollator_SSL, BalancedBatchSampler
from symbolic.symbolic import process_code


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

def train(output_dir_path, seed, params):
    train_losses = []
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

    if params['aug'] in ['forwhile', 'back-translation']:
        train_labeled_loader, dev_loader, test_loader, n_classes, train_dataset_l = get_dataloader_single(
            '../data/' + params['dataset'],params['dataset'], params['bs'], params['aug'],
            params['net_arch'])
    elif params['aug'] == 'all':
        train_labeled_loader, dev_loader, test_loader, n_classes, train_dataset_l = get_dataloader_all(
            '../data/' + params['dataset'],params['dataset'], params['bs'], params['net_arch'])
    else:
        train_labeled_loader, dev_loader, test_loader, n_classes, train_dataset_l = get_dataloader_sup(
            '../data/' + params['dataset'], params['bs'], params['net_arch'])


    print(n_classes)
    print(f"Complexity Class Number: {n_classes}")
    print(f"Initial Train Data Number: {len(train_dataset_l)}")
    print(f"Valid Data Number: {len(dev_loader)}")
    print(f"Test Data Number: {len(test_loader)}")

    # Initialize model
    netgroup = NetGroup(params['net_arch'], params['num_nets'], n_classes, device, params['lr'])
    netgroup.to(device)
    tokenizer = AutoTokenizer.from_pretrained(params['net_arch'])
    best_train_dataset_l = train_dataset_l

    # Load data
    
    best_checkpoint_acc_val = 0.0
    best_checkpoint_val_loss = 1000.0
    best_checkpoint_acc_test = 0.0
    best_checkpoint_epoch = 0.0
    best_checkpoint_f1_macro_test = 0.0
    best_conf_matrix = None

    pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
    for epoch in range(params["max_epoch"]):
        epoch_train_num = len(train_dataset_l)
        train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
        train_loss = train_one_epoch(netgroup, train_labeled_loader, device)
        val_loss = calculate_loss(netgroup, dev_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Evaluate
        acc_train, _, _ = evaluate(netgroup, train_labeled_loader, device)
        acc_val, _, _ = evaluate(netgroup, dev_loader, device)
        acc_test, f1_macro_test, conf_matrix = evaluate(netgroup, test_loader, device)

        training_stats.append(
            {   'epoch': epoch, #배치수
                'acc_train': acc_train,#train의 acc
                'acc_val': acc_val,#valid의 acc
                'acc_test': acc_test,#test의 acc
                'f1_test': f1_macro_test, #test의 f1 macro
            })


        if params["checkpoint"] == 'loss':
            if val_loss < best_checkpoint_val_loss:
                best_checkpoint_acc_test = acc_test
                best_checkpoint_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                best_conf_matrix = conf_matrix
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, "{}.{}".format(params["lr"],params["acc_save_name"])))
        elif params["checkpoint"] == 'acc':
            if acc_test > best_checkpoint_acc_test:
                best_checkpoint_acc_test = acc_test
                best_checkpoint_f1_macro_test = f1_macro_test
                best_checkpoint_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                best_conf_matrix = conf_matrix
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, "{}.{}".format(params["lr"],params["acc_save_name"])))
        
        pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                   f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1 Macro: {f1_macro_test:.4f}, ")
        pbar.update(1)

    pd.set_option('precision', 4)
    df_stats= pd.DataFrame(training_stats)
    df_stats = df_stats.set_index('epoch')
    training_stats_path = output_dir_path + 'training_statistics.csv'   
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
        plt.savefig(os.path.join(output_dir_path, "{}.{}.png".format(params['lr'],plot_type)), bbox_inches='tight')
        plt.close()  # Closes the plot

    # loss 값의 변화를 그래프로 시각화합니다.
    plt.figure(figsize=(20,14))
    #epochs = range(1, max_epoch + 1)  # epoch 수
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(output_dir_path+'{}.loss_plot.png'.format(params['lr']), bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir_path, "{}.confmat.png".format(params['lr'])), bbox_inches='tight')  # Saves the plot as a PNG file
    plt.close()  # Closes the plot

    pbar.write(f"(Valid {params['checkpoint']}) Best Epoch: {best_checkpoint_epoch}, Best Test Accuracy: {best_checkpoint_acc_test}, \
                Best Test F1 Macro: {best_checkpoint_f1_macro_test:.4f}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
    pbar.close()
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
    class_complexity_dict = {0: r'$O(1)$', 1: r'$O(N)$', 2: r'$O(\log N)$', 3: r'$O(N^2)$', 4: r'$O(N^3)$', 5: r'$O(N \log N)$', 6: r'$O(2^N)$'}
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
    if params['aug'] in ['forwhile', 'back-translation','all']:
        output_dir_path = './experiment/{}_{}_supervised_{}/'.format(params['dataset'], params['model_name'], params['aug'])
    else:
        output_dir_path = './experiment/{}_{}_supervised/'.format(params['dataset'], params['model_name'])
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Use the merged parameters for further processing
    print("Merged parameters:", params)

    # Train
    for i in range(3):
        seed = params['seed']+i
        if params['aug'] in ['forwhile', 'back-translation','all']:
            output_seed_path = './experiment/{}_{}_supervised_{}/{}'.format(params['dataset'], params['model_name'], params['aug'], seed)
        else:
            output_seed_path = './experiment/{}_{}_supervised/{}'.format(params['dataset'], params['model_name'], seed)
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
        'model name': params['model_name'],
        'augmentation': params['aug'],
        'Mean_std_acc': '%.2f \\pm %.2f' % (100*mean_acc, 100*st_dev_acc),
        'Mean_std_f1_macro': '%.2f \\pm %.2f' % (100*mean_f1_macro, 100*st_dev_f1_macro),
    }

    # Save best model info
    df = pd.DataFrame([final])
    csv_path = output_dir_path + 'summary_avgrun.csv'
    df.to_csv(csv_path, mode='a', index=False, header=True)
    print('\nSave best record in: ', csv_path)
    print("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description here')

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--checkpoint', type=str, help='retrieve the best valid loss checkpoint or valid acc checkpoint')
    parser.add_argument('--aug', type=str, help='augment setting: forwhile, back-translation, all, none')

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
