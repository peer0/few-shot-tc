import re
import ast
import csv
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TimeComplexityCalculator:
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(code)

    def extract_functions(self):
        functions = []
        lines = self.code.split('\n')
        current_function = ''
        for line in lines:
            if line.strip().startswith('def '):
                if current_function:
                    functions.append(current_function.strip())
                current_function = line
            else:
                current_function += '\n' + line
        if current_function:
            functions.append(current_function.strip())
        return functions

    def extract_loop_counts_and_function_calls(self, func_code):
        loop_counts = []
        function_calls = []
        lines = func_code.split('\n')
        for line in lines:
            if 'for' in line and 'range' in line:
                match = re.search(r'range\s*\(\s*(.*?)\s*\)', line)
                if match:
                    loop_counts.append(match.group(1).split(','))
            elif 'while' in line:
                match = re.search(r'while\s+(.*):', line)
                if match:
                    loop_counts.append(match.group(1).split(','))
            func_call_match = re.search(r'(\w+)\s*\(', line)
            if func_call_match and not line.strip().startswith('def '):
                function_calls.append(func_call_match.group(1))
        return loop_counts, function_calls

    def extract_loop_counts(self):
        loop_counts = []
        lines = self.code.split('\n')
        for line in lines:
            if 'for' in line and 'range' in line:
                match = re.search(r'range\s*\(\s*(.*?)\s*\)', line)
                if match:
                    loop_counts.append(match.group(1).split(','))
            elif 'while' in line:
                match = re.search(r'while\s+(.*):', line)
                if match:
                    loop_counts.append(match.group(1).split(','))
        return loop_counts

    def calculate_time_complexity_with_function_calls(self):
        functions = self.extract_functions()
        function_call_counts = {func: 0 for func in functions}
        function_loop_counts = {}
        function_call_names = {}

        for func_code in functions:
            match = re.search(r'def (\w+)\(', func_code)
            if not match:
                continue
            function_name = match.group(1)
            loop_counts, function_calls = self.extract_loop_counts_and_function_calls(func_code)
            function_loop_counts[function_name] = loop_counts
            function_call_names[function_name] = function_calls

        for func_name, calls in function_call_names.items():
            for call in calls:
                if call in function_call_counts:
                    function_call_counts[call] += 1  # 호출 횟수 갱신

        time_complexity_parts = []
        total_complexity = ""

        for func_name, loop_counts in function_loop_counts.items():
            time_complexity_part = []
            for count in loop_counts:
                time_complexity_part.append(" * ".join(count))
            for call_name in function_call_names.get(func_name, []):
                if call_name in function_call_counts:
                    time_complexity_part.append(f"{function_call_counts[call_name]}")
            time_complexity_parts.append("(" + " * ".join(time_complexity_part) + ")")

        # Check for logn and nlogn complexities
        logn_nlogn_complexity = self.detect_time_complexity_for_logn_nlogn()
        if logn_nlogn_complexity:
            time_complexity_part.append(logn_nlogn_complexity)
        
        total_complexity = " + ".join(time_complexity_parts) if time_complexity_parts else ""        
        
        return total_complexity

    def calculate_time_complexity(self):
        loop_counts = self.extract_loop_counts()
        time_complexity_parts = []
        total_complexity = ""

        for counts in loop_counts:
            time_part = ""
            for count in counts:
                time_part += count + " * "
            time_complexity_parts.append(time_part[:-3])  # Remove the last " * "

        if time_complexity_parts:
            total_complexity = " * ".join(time_complexity_parts)

        return total_complexity
    
    def detect_time_complexity_for_logn_nlogn(self):
        # Analyze the code for logarithmic and nlogn complexities
        analyzer = LoopAndRecursiveAnalysis()
        analyzer.visit(self.tree)
        
        if analyzer.contains_sort:
            return 'nlogn'
        elif analyzer.results:
            return 'logn'
        else:
            return ''

    def classify_time_complexity(self, time_complexity):
        complexity_parts = time_complexity.split(' + ')
        max_complexity_label = 1

        for part in complexity_parts:
            factors = part.split(' * ')
            char_count = 0

            for f in factors:
                if any(char.isalpha() for char in f):
                    char_count += 1
            
            if 'log' in part:
                complexity_label = 2 if 'n' not in part else 4
            elif char_count == 0:
                complexity_label = 1
            elif char_count == 1:
                complexity_label = 3
            elif char_count == 2:
                complexity_label = 5
            elif char_count == 3:
                complexity_label = 6
            else:
                complexity_label = 7

            max_complexity_label = max(max_complexity_label, complexity_label)

        return max_complexity_label

class LoopAndRecursiveAnalysis(ast.NodeVisitor):
    def __init__(self):
        self.results = []
        self.contains_sort = False

    def visit_While(self, node):
        for n in node.body:
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and isinstance(n.value, ast.BinOp):
                        if isinstance(n.value.op, ast.FloorDiv):
                            if (isinstance(n.value.right, ast.Constant) and
                                n.value.right.value == 2):
                                self.results.append(('While loop', node.lineno, target.id))
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['sort', 'sorted']:
                self.contains_sort = True
            for arg in node.args:
                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.FloorDiv):
                    if (isinstance(arg.right, ast.Constant) and
                        arg.right.value == 2):
                        self.results.append(('Recursive call', node.lineno, node.func.id))
        self.generic_visit(node)

def save_result(args):
    # Read the code from the dataset
    with open(args.dataset, 'r') as file:
        source_code = file.read()

    # Analyze the code complexity
    calculator = TimeComplexityCalculator(source_code)
    time_complexity = calculator.calculate_time_complexity_with_function_calls()
    complexity_classification = calculator.classify_time_complexity(time_complexity)

    # Print the results
    print("Time Complexity:", time_complexity)
    print("Complexity Classification:", complexity_classification)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze code for time complexity.")
    parser.add_argument('dataset', type=str, help="Path to the source code file to analyze.")
    args = parser.parse_args()
    save_result(args)
