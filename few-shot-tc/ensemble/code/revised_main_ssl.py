import os
import time
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from models.netgroup import NetGroup
from utils.helper import format_time
from utils.aug_dataloader import get_dataloader_v1,get_dataloader_v2,get_dataloader_v3, get_dataloader_v4
from criterions.criterions import ce_loss, consistency_loss
from utils.helper import freematch_fairness_loss
from utils.dataloader import MyCollator_SSL, BalancedBatchSampler


def pseudo_labeling(netgroup, train_dataset_l,train_unlabeled_loader, psl_threshold_h, psl_total, device, pbar):
    psl_correct = 0
    idx = 0
    train_dataset_u = train_unlabeled_loader.dataset
    for batch_unlabel in train_unlabeled_loader:
        origin_sent = train_dataset_u.sents[idx]
        answer_label = train_dataset_u.labels[idx]-1
        x_ulb_s = batch_unlabel['x']
        with torch.no_grad():
            outs_x_ulb_w_nets = netgroup.forward(x_ulb_s)
        logits_x_ulb_w = outs_x_ulb_w_nets[0]
        probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
        max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)
        assert batch_unlabel['label'].item() == answer_label
        #if max_probs > 0.6:
        #    pbar.write(f"Model confidence: {max_probs}, Predict label: {max_idx}, Reference label: {batch_unlabel['label'].item()}")
        for target_idx, target_probs in zip(max_idx, max_probs):
            target_idx = int(target_idx)
            if target_probs >= psl_threshold_h:
                psl_total += 1
                if target_idx == batch_unlabel['label'].item():
                    psl_correct += 1
                #import pdb; pdb.set_trace()
                # Update dataset with pseudo-label
                train_dataset_l.add_data(origin_sent, target_idx + 1)  # Add pseudo-labeled data to the labeled dataset
        idx+=1
    return train_dataset_l, psl_total, psl_correct

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

def train(output_dir_path, **params):
    torch.manual_seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psl_total = 0

    if params['version'] == 'v1':
        train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader_v1(
            '../data/' + params['dataset'],params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
            params['aug'],params['net_arch'])
        print(n_classes)
        print(f"Train Data Number: {len(train_labeled_loader)}")
        print(f"Valid Data Number: {len(dev_loader)}")
        print(f"Test Data Number: {len(test_loader)}")

        # Initialize model
        netgroup = NetGroup(params['net_arch'], params['num_nets'], n_classes, device, params['lr'])
        netgroup.to(device)
        tokenizer = AutoTokenizer.from_pretrained(params['net_arch'])
        best_train_dataset_l = train_dataset_l

        # Initialize optimizer
        #optimizer = torch.optim.Adam(netgroup.parameters(), lr=params['lr'])

        # Load data
        train_unlabeled_loader = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False,
                                            collate_fn=MyCollator_SSL(tokenizer))

        best_valacc_acc = 0.0
        best_valacc_loss = 0.0
        best_acc_testacc = 0.0
        best_loss_testacc = 0.0
        best_valloss_acc = 0.0
        best_valloss_loss = 0.0
        best_acc_model_epoch = 0
        best_loss_model_epoch = 0
        early_stop_count = 0
        pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
        for epoch in range(params["max_epoch"]):
            if early_stop_count >= params["early_stop_tolerance"]:
                print('Early stopping trigger at epoch:', epoch)
                break
            # Update train labeled loader with new pseudo-labeled data
            # train_labeled_loader.dataset.update_data(train_dataset_u)
            train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
            train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
            train_loss = train_one_epoch(netgroup, train_labeled_loader, device)
            val_loss = calculate_loss(netgroup, dev_loader, device)
            # Evaluate
            acc_train, _, _ = evaluate(netgroup, train_labeled_loader, device)
            acc_val, _, _ = evaluate(netgroup, dev_loader, device)
            acc_test, f1_macro_test, conf_matrix = evaluate(netgroup, test_loader, device)
            train_dataset_l, psl_total, psl_correct = pseudo_labeling(netgroup, train_dataset_l, train_unlabeled_loader, params['psl_threshold_h'], psl_total, device, pbar)
            
            if acc_val > best_valacc_acc:
                best_acc_testacc = acc_test
                best_valacc_acc = acc_val
                best_acc_model_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                early_stop_count = 0
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["acc_save_name"]))
            elif val_loss < best_valloss_loss:
                best_loss_testacc = acc_test
                best_valloss_loss = val_loss
                best_loss_model_epoch = epoch + 1
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["loss_save_name"]))
            else:
                early_stop_count += 1

            pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                    f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test macro_F1: {f1_macro_test:.4f}"
                    f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}, "
                    f"Train Data Number: {len(train_dataset_l)}")
            pbar.update(1)
        pbar.write(f"(Valid Acc) Best Epoch: {best_acc_model_epoch}, Best Valid Acc: {best_valacc_acc}, Best Test Accuracy: {best_acc_testacc}, Best Test F1(macro): {f1_macro_test}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.write(f"(Valid Loss) Best Epoch: {best_loss_model_epoch}, Best Valid Loss: {best_valloss_loss}, Best Test Accuracy: {best_loss_testacc}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.close()
        return best_acc_model_epoch, best_loss_model_epoch, best_acc_testacc, best_loss_testacc
        
    elif params['version'] == 'v2':
        train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader_v2(
            '../data/' + params['dataset'],params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
            params['aug'],params['net_arch'])
                
        print(n_classes)
        print(f"Train Data Number: {len(train_labeled_loader)}")
        print(f"Valid Data Number: {len(dev_loader)}")
        print(f"Test Data Number: {len(test_loader)}")

        # Initialize model
        netgroup = NetGroup(params['net_arch'], params['num_nets'], n_classes, device, params['lr'])
        netgroup.to(device)
        tokenizer = AutoTokenizer.from_pretrained(params['net_arch'])
        best_train_dataset_l = train_dataset_l

        # Initialize optimizer
        #optimizer = torch.optim.Adam(netgroup.parameters(), lr=params['lr'])

        # Load data
        train_unlabeled_loader = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False,
                                            collate_fn=MyCollator_SSL(tokenizer))

        best_valacc_acc = 0.0
        best_valacc_loss = 0.0
        best_acc_testacc = 0.0
        best_loss_testacc = 0.0
        best_valloss_acc = 0.0
        best_valloss_loss = 0.0
        best_acc_model_epoch = 0
        best_loss_model_epoch = 0
        early_stop_count = 0
        pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
        for epoch in range(params["max_epoch"]):
            if early_stop_count >= params["early_stop_tolerance"]:
                print('Early stopping trigger at epoch:', epoch)
                break
            # Update train labeled loader with new pseudo-labeled data
            # train_labeled_loader.dataset.update_data(train_dataset_u)
            train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
            train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
            train_loss = train_one_epoch(netgroup, train_labeled_loader, device)
            val_loss = calculate_loss(netgroup, dev_loader, device)
            # Evaluate
            acc_train, _, _ = evaluate(netgroup, train_labeled_loader, device)
            acc_val, _, _ = evaluate(netgroup, dev_loader, device)
            acc_test, f1_macro_test, conf_matrix = evaluate(netgroup, test_loader, device)
            train_dataset_l, psl_total, psl_correct = pseudo_labeling(netgroup, train_dataset_l, train_unlabeled_loader, params['psl_threshold_h'], psl_total, device, pbar)
            
            if acc_val > best_valacc_acc:
                best_acc_testacc = acc_test
                best_valacc_acc = acc_val
                best_acc_model_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                early_stop_count = 0
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["acc_save_name"]))
            elif val_loss < best_valloss_loss:
                best_loss_testacc = acc_test
                best_valloss_loss = val_loss
                best_loss_model_epoch = epoch + 1
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["loss_save_name"]))
            else:
                early_stop_count += 1

            pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                    f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test macro_F1: {f1_macro_test:.4f}"
                    f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}, "
                    f"Train Data Number: {len(train_dataset_l)}")
            pbar.update(1)
        pbar.write(f"(Valid Acc) Best Epoch: {best_acc_model_epoch}, Best Valid Acc: {best_valacc_acc}, Best Test Accuracy: {best_acc_testacc}, Best Test F1(macro): {f1_macro_test},Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.write(f"(Valid Loss) Best Epoch: {best_loss_model_epoch}, Best Valid Loss: {best_valloss_loss}, Best Test Accuracy: {best_loss_testacc}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.close()
        return best_acc_model_epoch, best_loss_model_epoch, best_acc_testacc, best_loss_testacc, best_valacc_acc, f1_macro_test, best_train_dataset_l, conf_matrix

    elif params['version'] == 'v3':
        train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader_v3(
            '../data/' + params['dataset'],params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
            params['aug'],params['net_arch'])
        print(n_classes)
        print(f"Train Data Number: {len(train_labeled_loader)}")
        print(f"Valid Data Number: {len(dev_loader)}")
        print(f"Test Data Number: {len(test_loader)}")

        # Initialize model
        netgroup = NetGroup(params['net_arch'], params['num_nets'], n_classes, device, params['lr'])
        netgroup.to(device)
        tokenizer = AutoTokenizer.from_pretrained(params['net_arch'])
        best_train_dataset_l = train_dataset_l

        # Initialize optimizer
        #optimizer = torch.optim.Adam(netgroup.parameters(), lr=params['lr'])

        # Load data
        train_unlabeled_loader = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False,
                                            collate_fn=MyCollator_SSL(tokenizer))

        best_valacc_acc = 0.0
        best_valacc_loss = 0.0
        best_acc_testacc = 0.0
        best_loss_testacc = 0.0
        best_valloss_acc = 0.0
        best_valloss_loss = 0.0
        best_acc_model_epoch = 0
        best_loss_model_epoch = 0
        early_stop_count = 0
        pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
        for epoch in range(params["max_epoch"]):
            if early_stop_count >= params["early_stop_tolerance"]:
                print('Early stopping trigger at epoch:', epoch)
                break
            # Update train labeled loader with new pseudo-labeled data
            # train_labeled_loader.dataset.update_data(train_dataset_u)
            train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
            train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
            train_loss = train_one_epoch(netgroup, train_labeled_loader, device)
            val_loss = calculate_loss(netgroup, dev_loader, device)
            # Evaluate
            acc_train, _, _ = evaluate(netgroup, train_labeled_loader, device)
            acc_val, _, _ = evaluate(netgroup, dev_loader, device)
            acc_test, f1_macro_test, conf_matrix = evaluate(netgroup, test_loader, device)
            train_dataset_l, psl_total, psl_correct = pseudo_labeling(netgroup, train_dataset_l, train_unlabeled_loader, params['psl_threshold_h'], psl_total, device, pbar)
            
            if acc_val > best_valacc_acc:
                best_acc_testacc = acc_test
                best_valacc_acc = acc_val
                best_acc_model_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                early_stop_count = 0
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["acc_save_name"]))
            elif val_loss < best_valloss_loss:
                best_loss_testacc = acc_test
                best_valloss_loss = val_loss
                best_loss_model_epoch = epoch + 1
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["loss_save_name"]))
            else:
                early_stop_count += 1

            pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                    f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test macro_F1: {f1_macro_test:.4f}"
                    f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}, "
                    f"Train Data Number: {len(train_dataset_l)}")
            pbar.update(1)
        pbar.write(f"(Valid Acc) Best Epoch: {best_acc_model_epoch}, Best Valid Acc: {best_valacc_acc}, Best Test Accuracy: {best_acc_testacc}, Best Test F1(macro): {f1_macro_test}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.write(f"(Valid Loss) Best Epoch: {best_loss_model_epoch}, Best Valid Loss: {best_valloss_loss}, Best Test Accuracy: {best_loss_testacc}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.close()
        return best_acc_model_epoch, best_loss_model_epoch, best_acc_testacc, best_loss_testacc, conf_matrix

    elif params['version'] == 'v4':
        train_labeled_loader, _, dev_loader, test_loader, n_classes, train_dataset_l, train_dataset_u = get_dataloader_v4(
            '../data/' + params['dataset'],params['dataset'], params['n_labeled_per_class'], params['bs'], params['load_mode'],
            params['aug'],params['net_arch'])
                
        print(n_classes)
        print(f"Train Data Number: {len(train_labeled_loader)}")
        print(f"Valid Data Number: {len(dev_loader)}")
        print(f"Test Data Number: {len(test_loader)}")

        # Initialize model
        netgroup = NetGroup(params['net_arch'], params['num_nets'], n_classes, device, params['lr'])
        netgroup.to(device)
        tokenizer = AutoTokenizer.from_pretrained(params['net_arch'])
        best_train_dataset_l = train_dataset_l

        # Initialize optimizer
        #optimizer = torch.optim.Adam(netgroup.parameters(), lr=params['lr'])

        # Load data
        train_unlabeled_loader = DataLoader(dataset=train_dataset_u, batch_size=1, shuffle=False,
                                            collate_fn=MyCollator_SSL(tokenizer))

        best_valacc_acc = 0.0
        best_valacc_loss = 0.0
        best_acc_testacc = 0.0
        best_loss_testacc = 0.0
        best_valloss_acc = 0.0
        best_valloss_loss = 0.0
        best_acc_model_epoch = 0
        best_loss_model_epoch = 0
        early_stop_count = 0
        pbar = tqdm(total=params["max_epoch"], desc="Training", position=0, leave=True)
        for epoch in range(params["max_epoch"]):
            if early_stop_count >= params["early_stop_tolerance"]:
                print('Early stopping trigger at epoch:', epoch)
                break
            # Update train labeled loader with new pseudo-labeled data
            # train_labeled_loader.dataset.update_data(train_dataset_u)
            train_sampler = BalancedBatchSampler(train_dataset_l,params['bs'])
            train_labeled_loader = DataLoader(dataset=train_dataset_l, batch_size=params['bs'], sampler=train_sampler, collate_fn=MyCollator_SSL(tokenizer))
            train_loss = train_one_epoch(netgroup, train_labeled_loader, device)
            val_loss = calculate_loss(netgroup, dev_loader, device)
            # Evaluate
            acc_train, _, _ = evaluate(netgroup, train_labeled_loader, device)
            acc_val, _, _ = evaluate(netgroup, dev_loader, device)
            acc_test, f1_macro_test, conf_matrix = evaluate(netgroup, test_loader, device)
            train_dataset_l, psl_total, psl_correct = pseudo_labeling(netgroup, train_dataset_l, train_unlabeled_loader, params['psl_threshold_h'], psl_total, device, pbar)
            
            if acc_val > best_valacc_acc:
                best_acc_testacc = acc_test
                best_valacc_acc = acc_val
                best_acc_model_epoch = epoch + 1
                best_train_dataset_l = train_dataset_l
                early_stop_count = 0
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["acc_save_name"]))
            elif val_loss < best_valloss_loss:
                best_loss_testacc = acc_test
                best_valloss_loss = val_loss
                best_loss_model_epoch = epoch + 1
                torch.save(netgroup.state_dict(), os.path.join(output_dir_path, params["loss_save_name"]))
            else:
                early_stop_count += 1

            pbar.write(f"Epoch {epoch + 1}/{params['max_epoch']}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Train Acc: {acc_train:.4f}, "
                    f"Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}, Test macro_F1: {f1_macro_test:.4f}"
                    f"Total Pseudo-Labels: {psl_total}, Correct Pseudo-Labels: {psl_correct}, "
                    f"Train Data Number: {len(train_dataset_l)}")
            pbar.update(1)
        pbar.write(f"(Valid Acc) Best Epoch: {best_acc_model_epoch}, Best Valid Acc: {best_valacc_acc}, Best Test Accuracy: {best_acc_testacc}, Best Test F1(macro): {f1_macro_test},Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.write(f"(Valid Loss) Best Epoch: {best_loss_model_epoch}, Best Valid Loss: {best_valloss_loss}, Best Test Accuracy: {best_loss_testacc}, Best Pseudo-Labeled Number: {len(best_train_dataset_l)}\n")
        pbar.close()
        return best_acc_model_epoch, best_loss_model_epoch, best_acc_testacc, best_loss_testacc, best_valacc_acc, f1_macro_test, best_train_dataset_l, conf_matrix


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
    
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro'), confusion_matrix(all_labels, all_preds)

def main(config_file='config.json', **kwargs):
    # Load parameters from config file
    params = load_config(config_file)

    # Override parameters from config file with command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    output_dir_path = './experiment/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}_{}/'.format(params['model'],params['seed'],params['aug'],params['version'],params['dataset'], params['model_name'], params['n_labeled_per_class'],params['psl_threshold_h'],params['lr'],params['seed'],params['aug'],params['version'])
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Use the merged parameters for further processing
    print("Merged parameters:", params)

    # Train
    best_acc_model_epoch, best_loss_model_epoch, best_acc_testacc, best_loss_testacc, best_valacc_acc,f1_macro_test, best_train_dataset_l, conf_matrix = train(output_dir_path, **params)

    label_dict = {0: 'constant', 1: 'logn', 2: 'linear', 3: 'nlogn', 4: 'quadratic', 5: 'cubic', 6: 'exponential'}

    # conf_matrix를 DataFrame으로 변환합니다.
    conf_matrix_df = pd.DataFrame(conf_matrix)

    # 인덱스와 열 이름을 실제 레이블로 변경합니다.
    conf_matrix_df.columns = [label_dict[i] for i in range(conf_matrix_df.shape[1])]
    conf_matrix_df.index = [label_dict[i] for i in range(conf_matrix_df.shape[0])]

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    output_file = os.path.join(output_dir_path, 'confusion_matrix.png')
    plt.savefig(output_file, bbox_inches='tight')  # Saves the plot as a PNG file
    plt.close()  # Closes the plot to prevent it from displaying in the notebook

    
    # Save best model info
    with open(os.path.join(output_dir_path, "best_model_info.txt"), "w") as f:
        f.write(f"(Valid Acc) Best Epoch: {best_acc_model_epoch}, Best Valid Acc: {best_valacc_acc}, Best Test Accuracy: {best_acc_testacc}, Best Test F1(macro): {f1_macro_test},Best Pseudo-Labeled Number: {len(best_train_dataset_l)}")
        # f.write(f"(Valid Loss) Best Epoch: {best_loss_model_epoch}, Best Accuracy: {best_loss_testacc}")

    
    print("Training complete!")


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description here')

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--n_labeled_per_class', type=int, required=True, help='Number of labeled samples per class')
    parser.add_argument('--psl_threshold_h', type=float, required=True, help='Threshold for pseudo-labeling')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--aug', type=str, help='aug Dataset type')
    parser.add_argument('--version', type=str, help='aug experiment type')
    parser.add_argument('--model', type=str, help='model type')
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
