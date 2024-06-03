import re
import ast
import csv
import json
import argparse
import numpy as np
import os
from os import walk
import sys
from time import sleep
import javalang
from javalang import parse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from symbolic.calculator import CodeComplexityCalculator


def process_code(code, language):
    if language == 'corcod': language = 'java'
    calculator = CodeComplexityCalculator(code, language)
    if language == 'java':
        time_complexity = calculator.calculate_time_complexity_with_function_calls()
        prediction = calculator.classify_time_complexity(time_complexity)

    elif language == 'python':
        if "def " in code:
            time_complexity = calculator.calculate_time_complexity_with_function_calls()
        else:
            time_complexity = calculator.calculate_time_complexity()
        prediction = calculator.classify_time_complexity(time_complexity)
    #return time_complexity, prediction, calculator.error_method, calculator.success_method
    return prediction


def save_result(src_csv_path, dest_csv_path, language):
    error_code = 0
    success_code = 0
    error_methods = 0
    success_methods = 0
    y_true = []
    y_pred = []
    with open(dest_csv_path, mode='w', newline='', encoding='utf-8') as dest_file:
        writer = csv.writer(dest_file)
        writer.writerow(['src', 'ground_truth', 'predicted_tc'])
        with open(src_csv_path, newline='', encoding='utf-8') as src_file:
            reader = csv.DictReader(src_file)
            for row in reader:
                src_code = row['content']
                ground_truth_label = int(row['label'])
                #predicted_label = process_code(src_code, language, ground_truth_label)
                complexity_prediction, prediction, error_method, success_method = process_code(src_code, language)
                if len(complexity_prediction) > 0: success_code += 1
                else: error_code += 1
                error_methods += error_method
                success_methods += success_method
                writer.writerow([src_code, ground_truth_label, prediction])
                y_true.append(ground_truth_label)
                y_pred.append(prediction)
    print(f"code success: {success_code}, code error: {error_code}, method success: {success_methods}, method error: {error_methods}")
    
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate Micro-F1 and Macro-F1 scores
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    classes = np.unique(np.concatenate((y_true, y_pred)))
    classes = list(map(int, classes))
    classwise_accuracy = {}
    for cls in classes:
        class_y_pred = []
        class_y_true = []
        indices = []
        for i in range(len(y_true)):
            if y_true[i] == cls:
                indices.append(i)
        for i in indices:
            class_y_pred.append(y_pred[i])
            class_y_true.append(y_true[i])

        class_accuracy = accuracy_score(class_y_true, class_y_pred)
        classwise_accuracy[cls] = class_accuracy


    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{args.language} Confusion Matrix')
    output_file = 'confusion_matrix_java_temp.png'
    plt.savefig(output_file, bbox_inches='tight')  # Saves the plot as a PNG file

    print(f"Accuracy: {accuracy}")
    print(f"Macro-F1 Score: {macro_f1}")
    print("Class-wise Accuracy:")
    for cls, acc in classwise_accuracy.items():
        print(f"{cls}: {acc}")


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Calculate and save code complexity.")
#    parser.add_argument("--dataset", default=".\\data\\int_label\\corcod\\test.csv", help="Path to the source CSV file")
#    parser.add_argument("--result", default=".\\result\\temp_result.csv", help="Path to the destination CSV file to save results")
#    parser.add_argument("--language", default="java", help="Programming language of the source code (python/java)")
#
#    args = parser.parse_args()
#
#    save_result(args.dataset, args.result, args.language)
#
