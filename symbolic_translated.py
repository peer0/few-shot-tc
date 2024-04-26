# Translated code from 'symbolic_JH.py' by GPT-4
# i.e., computing java code complexity --> computing python code complexity
# hsan

import ast
import csv
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class PythonCodeComplexityCalculator:
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(self.code)
        self.success_method = 0
        self.error_method = 0

    def extract_functions(self):
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                function_info = {
                    "name": node.name,
                    "body": node.body,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                }
                functions.append(function_info)
        return functions

    def analyze_function(self, function_info):
        visitor = ComplexityVisitor()
        visitor.visit(ast.Module(body=function_info['body']))
        return visitor.complexity

    def calculate_time_complexity_with_function_calls(self):
        functions = self.extract_functions()
        function_complexities = []
        for function in functions:
            try:
                complexity = self.analyze_function(function)
                function_complexities.append(complexity)
                self.success_method += 1
            except Exception as e:
                self.error_method += 1
                function_complexities.append('')

        total_complexity = " + ".join(function_complexities)
        return total_complexity

    def classify_time_complexity(self, time_complexity):
        # More nuanced classification based on string patterns
        complexity_terms = time_complexity.split(' + ')
        max_complexity_label = 'constant'  # Default is constant
        for term in complexity_terms:
            if 'n^2' in term or 'n * n' in term:
                max_complexity_label = 'quadratic'
            elif 'n^3' in term:
                max_complexity_label = 'cubic'
            elif 'n * logn' in term or 'logn * n' in term:
                max_complexity_label = 'nlogn'
            elif 'logn' in term:
                max_complexity_label = 'logn'
            elif 'n' in term:
                max_complexity_label = 'linear'
            elif '2^n' in term or 'n!' in term:
                max_complexity_label = 'np'  # Exponential or factorial
        return max_complexity_label

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = []

    def visit_For(self, node):
        self.complexity.append('n')
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity.append('n')
        self.generic_visit(node)

    def visit_If(self, node):
        self.complexity.append('1')
        self.generic_visit(node)

    def visit_Call(self, node):
        self.complexity.append('1')
        self.generic_visit(node)

    def get_complexity(self):
        if not self.complexity:
            return 'constant'
        return ' + '.join(set(self.complexity))  # Simplify by removing duplicate terms

def process_code(code):
    calculator = PythonCodeComplexityCalculator(code)
    time_complexity = calculator.calculate_time_complexity_with_function_calls()
    complexity_label = calculator.classify_time_complexity(time_complexity)
    return time_complexity, complexity_label, calculator.error_method, calculator.success_method

def save_result(src_csv_path, dest_csv_path):
    with open(src_csv_path, newline='', encoding='utf-8') as src_file, open(dest_csv_path, mode='w', newline='', encoding='utf-8') as dest_file:
        reader = csv.DictReader(src_file)
        writer = csv.writer(dest_file)
        writer.writerow(['src', 'predicted_complexity', 'predicted_label'])

        for row in reader:
            src_code = row['content']
            complexity, label, error_method, success_method = process_code(src_code)
            writer.writerow([src_code, complexity, label])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Python code complexity.")
    parser.add_argument("--src_path", default="", help="Path to the source CSV file")
    parser.add_argument("--dest_path", default="", help="Path to the destination CSV file to save results")

    args = parser.parse_args()
    save_result(args.src_path, args.dest_path)
