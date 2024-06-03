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
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

from models.netgroup import NetGroup
from utils.helper import format_time
from utils.dataloader import get_dataloader_sup
from criterions.criterions import ce_loss, consistency_loss
from utils.helper import freematch_fairness_loss
from utils.dataloader import MyCollator_SSL, BalancedBatchSampler

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
    if params['dataset'] =='python':
        language = 'python'
    elif params['dataset'] =='java':
        language = 'java'
    elif params['dataset'] =='corcod':
        language = 'corcod'
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labeled_loader, dev_loader, test_loader, n_classes, train_dataset_l = get_dataloader_sup(
        '../data/' + params['dataset'],params['bs'], params['load_mode'],
        params['net_arch'])
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

    best_checkpoint_acc_val = 0.0
    best_checkpoint_val_loss = 1000.0
    best_checkpoint_acc_test = 0.0
    best_checkpoint_epoch = 0.0
    min_length = 0
    labels_with_min_length = 0
    best_checkpoint_f1_macro_test = 0.0

    pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
    for epoch in range(params["max_epoch"]):
        epoch_train_num = len(train_dataset_l)
        train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
        train_loss = train_one_epoch(netgroup, train_labeled_loader, device)
        val_loss = calculate_loss(netgroup, dev_loader, device)
        # Evaluate
        acc_train, _ = evaluate(netgroup, train_labeled_loader, device)
        acc_val, _ = evaluate(netgroup, dev_loader, device)
        acc_test, f1_macro_test = evaluate(netgroup, test_loader, device)

        if params["checkpoint"] == 'loss':
            if val_loss < best_checkpoint_val_loss:
                best_checkpoint_acc_test = acc_test
                best_checkpoint_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["acc_save_name"]))
        elif params["checkpoint"] == 'acc':
            if acc_test > best_checkpoint_acc_test:
                best_checkpoint_acc_test = acc_test
                best_checkpoint_f1_macro_test = f1_macro_test
                best_checkpoint_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["acc_save_name"]))

        pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                   f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1 Macro: {f1_macro_test:.4f}, "
                   )
        pbar.update(1)
    pbar.write(f"(Valid {params['checkpoint']}) Best Epoch: {best_checkpoint_epoch}, Best Test Accuracy: {best_checkpoint_acc_test}, \
                Best Test F1 Macro: {best_checkpoint_f1_macro_test:.4f}\n")
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
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

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
    output_dir_path = './experiment/{}_{}_supervised/'.format(params['dataset'], params['model_name'])
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Use the merged parameters for further processing
    print("Merged parameters:", params)

    # Train
    for i in range(3):
        seed = params['seed']+i
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
