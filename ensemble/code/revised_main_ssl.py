import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from utils.dataloader import get_dataloader
from models.netgroup import NetGroup
from utils.helper import format_time
from utils.dataloader import MyCollator_SSL, BalancedBatchSampler
import json

def pseudo_labeling(netgroup, train_dataset_l,train_unlabeled_loader, psl_threshold_h, psl_total, device):
    psl_correct = 0
    for batch_unlabel in train_unlabeled_loader:
        x_ulb_s = batch_unlabel['x']
        with torch.no_grad():
            outs_x_ulb_w_nets = netgroup.forward(x_ulb_s)
        logits_x_ulb_w = outs_x_ulb_w_nets[0]
        probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
        max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
        for target_idx, target_probs in zip(max_idx, max_probs):
            target_idx = int(target_idx)
            if target_probs >= psl_threshold_h:
                psl_total += 1
                if target_idx == batch_unlabel['label'].item() - 1:
                    psl_correct += 1
                # Update dataset with pseudo-label
                train_dataset_l.add_data(batch_unlabel['x'], target_idx + 1)  # Add pseudo-labeled data to the labeled dataset
    return train_dataset_l, psl_total, psl_correct

def calculate_loss(netgroup, data_loader, device):
    netgroup.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            outputs = netgroup.forward(inputs, labels)[0]
            loss = [ce_loss(outputs, labels)]
            total_loss += loss[0].item()
    return total_loss / len(data_loader)

def train_one_epoch(netgroup, train_labeled_loader, optimizer, device):
    netgroup.train()
    total_loss = 0.0
    for batch_label in train_labeled_loader:
        x_lb, y_lb = batch_label['x'], batch_label['label'].to(device)
        outs_x_lb = netgroup.forward(x_lb, y_lb)[0]
        sup_loss_nets = [ce_loss(outs_x_lb, y_lb)]
        optimizer.zero_grad()
        sup_loss_nets[0].backward()
        optimizer.step()
        total_loss += sup_loss_nets[0].item()
    return total_loss / len(train_labeled_loader)

def train(output_dir_path, **params):
    torch.manual_seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psl_total = 0

    train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader(
        root + 'data/' + params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
        params['net_arch'])

    # Initialize model
    netgroup = NetGroup(params['net_arch'], params['num_nets'], n_classes, device, params['lr'])
    netgroup.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(netgroup.parameters(), lr=params['lr'])

    # Load data
    train_unlabeled_loader = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False,
                                        collate_fn=MyCollator_SSL(tokenizer))

    best_acc = 0.0
    best_model_step = 0
    early_stop_count = 0
    pbar = tqdm(total=max_epoch, desc="Training", position=0, leave=True)
    for epoch in range(max_epoch):
        if early_stop_count >= early_stop_tolerance:
            print('Early stopping trigger at epoch:', epoch)
            break
        # Update train labeled loader with new pseudo-labeled data
        # train_labeled_loader.dataset.update_data(train_dataset_u)
        train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], shuffle=True, collate_fn=MyCollator_SSL(tokenizer))
        train_loss = train_one_epoch(netgroup, train_labeled_loader, optimizer, device)
        val_loss = calculate_loss(netgroup, val_loader, device)
        # Evaluate
        acc_train, _ = evaluate(netgroup, train_labeled_loader, device)
        acc_val, _ = evaluate(netgroup, dev_loader, device)
        if acc_val > best_acc:
            best_acc = acc_val
            best_model_epoch = epoch + 1
            early_stop_count = 0
            torch.save(netgroup.state_dict(), os.path.join(output_dir_path, f"{save_name}_best.pth"))
        else:
            early_stop_count += 1
        acc_test, f1_test = evaluate(netgroup, test_loader, device)
        train_dataset_l, psl_total, psl_correct = pseudo_labeling(netgroup, train_dataset_l, train_unlabeled_loader, params['psl_threshold_h'], psl_total, device)

        pbar.write(f"Epoch {epoch + 1}/{max_epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f},"
                   f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test F1: {f1_test:.4f}")
        pbar.update(1)
    pbar.close()
    return best_model_epoch, best_acc


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

    # Override parameters from config file with command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    output_dir_path = './experiment/{}_{}_{}_{}_{}_{}/'.format(params['dataset'], params['model_name'], params['n_labeled_per_class'],params['psl_threshold_h'],params['lr'],params['seed'])

    # Use the merged parameters for further processing
    print("Merged parameters:", params)

    # Train
    best_step, best_acc = train(output_dir_path, params)

    # Save best model info
    with open(os.path.join(output_dir_path, "best_model_info.txt"), "w") as f:
        f.write(f"Best Step: {best_step}\nBest Accuracy: {best_acc}")

    print("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description here')

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--n_labeled_per_class', type=int, required=True, help='Number of labeled samples per class')
    parser.add_argument('--psl_threshold_h', type=float, required=True, help='Threshold for pseudo-labeling')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Dataset name')

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
