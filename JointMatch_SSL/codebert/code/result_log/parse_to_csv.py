import csv
import os
import sys

import os
import re

# Function to extract information from a log text
def extract_info(log_text):
    # Extracting Best_step, best_test_acc, and best_test_f1
    best_step_match = re.search(r'Best_step:\s+(\d+)', log_text)
    best_test_acc_match = re.search(r'best_test_acc:\s+([\d.]+)', log_text)
    best_test_f1_match = re.search(r'best_test_f1:\s+([\d.]+)', log_text)

    if best_step_match and best_test_acc_match and best_test_f1_match:
        best_step = int(best_step_match.group(1))
        best_test_acc = float(best_test_acc_match.group(1))
        best_test_f1 = float(best_test_f1_match.group(1))

        # Extracting acc_val corresponding to Best_step
        step_line_match = re.search(r'.+Epoch\s+([\d.]+)\s+Step\s+' + str(best_step) + r'.+acc_val\s+([\d.]+)', log_text)
        #step_line_match = re.search(r'Step\s+' + str(best_step) + r'.+acc_val\s+([\d.]+)', log_text)
        if step_line_match:
            best_epoch = int(step_line_match.group(1))
            acc_val = float(step_line_match.group(2))
            return best_epoch, best_test_acc, best_test_f1, acc_val
        else:
            return None
    else:
        return None

# Iterate over all .log files in the current directory
results = []
for filename in os.listdir('.'):
    if filename.endswith('.log'):
        with open(filename, 'r') as file:
            log_text = file.read()
            result = extract_info(log_text)
            if result:
                fileinfo = filename.split('_')
                nshot = fileinfo[0]
                modelname = fileinfo[1]
                learning_rate = fileinfo[2]
                threshold = fileinfo[3]
                tolerence = fileinfo[4]
                datasetname = fileinfo[5]
                best_epoch, best_test_acc, best_test_f1, acc_val = result


                results.append((modelname,nshot,learning_rate,threshold,tolerence,acc_val, best_test_acc, best_test_f1, best_epoch))
                #print(results)
                #best_step, best_test_acc, best_test_f1, acc_val = result
                #print("File:", filename)
                #print("Best Step:", best_step)
                #print("Best Test Accuracy:", best_test_acc)
                #print("Best Test F1 Score:", best_test_f1)
                #print("acc_val at Best Step:", acc_val)
                #print()
                #input()
            else:
                print("No valid information found in", filename)
                print(log_text)
                input()

csv_filename = 'extracted_results.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['modelname', 'nshot', 'learning_rate', 'threshold', 'tolerence', 'acc_val', 'test_acc', 'test_f1', 'best_epoch'])
    for result in results:
        csv_writer.writerow(result)

print("Results saved to", csv_filename)
