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
        self.function_complexities = {}

    def extract_functions(self):
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        return functions

    def extract_loop_counts_and_function_calls(self, func_code):
        loop_counts = []
        function_calls = []

        def visit_node(node):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                loop_counts.append(node)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and not isinstance(node, ast.FunctionDef):
                    function_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute) and not isinstance(node, ast.FunctionDef):
                    function_calls.append(node.func.attr)
            for child in ast.iter_child_nodes(node):
                visit_node(child)

        visit_node(func_code)
        return loop_counts, function_calls

    def extract_loop_counts(self):
        loop_counts = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                loop_counts.append(node)
        return loop_counts

    def calculate_function_complexity(self, loop_counts):
        time_complexity_part = []
        nested_complexities = []

        for counts in loop_counts:
            depth = self.get_loop_depth(counts)
            complexity = self.get_complexity_of_loop(counts)
            if depth == 1:
                time_complexity_part.append(complexity)
            else:
                nested_complexities.append((depth, complexity, counts))

        # Process nested complexities
        nested_complexities.sort(reverse=True, key=lambda x: x[0])  # Sort by depth in descending order
        
        processed_parents = set()
        
        for depth, complexity, loop in nested_complexities:
            parent_index = self.find_parent_index(loop_counts, loop)
            if parent_index is not None and (parent_index, depth) not in processed_parents:
                if len(time_complexity_part) > parent_index:
                    time_complexity_part[parent_index] += " * " + complexity
                else:
                    time_complexity_part.append(complexity)
                processed_parents.add((parent_index, depth))
        
        return time_complexity_part


    def find_parent_index(self, loop_counts, target_node):
        for i, loop in enumerate(loop_counts):
            if self.is_parent(loop, target_node):
                return i
        return None

    def is_parent(self, parent, node):
        current_node = node
        while current_node:
            if current_node == parent:
                return True
            current_node = self.get_parent_node(current_node)
        return False

    def get_loop_depth(self, loop_node):
        depth = 1
        current_node = loop_node
        while current_node:
            parent = self.get_parent_node(current_node)
            if isinstance(parent, (ast.For, ast.While)):
                depth += 1
            current_node = parent
        return depth
    
    def get_parent_node(self, node):
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                if child is node:
                    return parent
        return None
        
    def get_complexity_of_loop(self, loop_node):
        if isinstance(loop_node, ast.For):
            if isinstance(loop_node.iter, ast.Call) and isinstance(loop_node.iter.func, ast.Name) and loop_node.iter.func.id == 'range':
                args = loop_node.iter.args
                if all(self.is_numeric_constant(arg) for arg in args):
                    return '1'
                if len(args) == 1:
                    return 'n'
                elif len(args) == 2:
                    return 'n'
                elif len(args) == 3:
                    return 'n'
        elif isinstance(loop_node, ast.While):
            if self.has_floor_division(loop_node) or self.has_bit_op(loop_node):
                return 'logn'
            condition = loop_node.test
            if isinstance(condition, ast.Compare):
                return 'logn'
            return 'n'
        return '1'
    
    def is_numeric_constant(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return True
        return False

    def has_floor_division(self, node):
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.BinOp) and isinstance(sub_node.op, ast.FloorDiv):
                return True
        return False
    
    def has_bit_op(self, node):
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.AugAssign) and isinstance(sub_node.op, (ast.RShift, ast.LShift)):
                return True
        return False

    def calculate_time_complexity_with_function_calls(self):
        functions = self.extract_functions()
        function_call_counts = {func.name: 0 for func in functions}
        function_loop_counts = {}
        function_call_names = {}

        for func_code in functions:
            function_name = func_code.name
            loop_counts, function_calls = self.extract_loop_counts_and_function_calls(func_code)
            function_loop_counts[function_name] = loop_counts
            function_call_names[function_name] = function_calls

        # Track function calls and calculate complexity only for called functions
        called_functions = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in function_call_counts:
                    called_functions.add(node.func.id)

        for func_name, loop_counts in function_loop_counts.items():
            if func_name in called_functions:
                self.function_complexities[func_name] = self.calculate_function_complexity(loop_counts)

        total_complexity = self.calculate_overall_complexity(function_call_counts, function_call_names, called_functions)
        
        # print("Total Complexity: ", total_complexity)

        return total_complexity

    def calculate_overall_complexity(self, function_call_counts, function_call_names, called_functions):
        time_complexity_parts = []
        global_complexity_parts = self.calculate_global_loop_complexity()

        for func_name, complexities in self.function_complexities.items():
            time_complexity_part = []
            for complexity in complexities:
                time_complexity_part.append(complexity)
            for call_name in function_call_names.get(func_name, []):
                if call_name in function_call_counts and call_name in called_functions:
                    if call_name in self.function_complexities:
                        call_complexity = " * ".join(self.function_complexities[call_name])
                        time_complexity_part.append(call_complexity)
            if time_complexity_part:
                time_complexity_parts.append("(" + " + ".join(time_complexity_part) + ")")

        if global_complexity_parts:
            time_complexity_parts.extend(global_complexity_parts)

        logn_nlogn_complexity = self.detect_time_complexity_for_logn_nlogn()
        if logn_nlogn_complexity:
            time_complexity_parts.append(logn_nlogn_complexity)
        
        # built_in_complexity = self.detect_time_complexity_for_built_in_func()
        # if built_in_complexity and not time_complexity_parts:
        #     time_complexity_parts.append(built_in_complexity)

        total_complexity = " + ".join(time_complexity_parts) if time_complexity_parts else ""

        return total_complexity

    def calculate_global_loop_complexity(self):
        global_loop_counts = self.extract_loop_counts()
        global_complexity_parts = []

        for loop in global_loop_counts:
            loop_complexity = self.get_complexity_of_loop(loop)
            calls_in_loop = self.get_function_calls_in_loop(loop)
            if calls_in_loop:
                for call in calls_in_loop:
                    if call in self.function_complexities:
                        call_complexity = " * ".join(self.function_complexities[call])
                        global_complexity_parts.append(f"{loop_complexity} * {call_complexity}")
            else:
                global_complexity_parts.append(loop_complexity)

        return global_complexity_parts

    def get_function_calls_in_loop(self, loop_node):
        calls = []

        def visit_node(node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
            for child in ast.iter_child_nodes(node):
                visit_node(child)

        visit_node(loop_node)
        return calls

    def detect_time_complexity_for_logn_nlogn(self):
        for_loop_count = 0
        has_sorting_or_bisect = False
        has_log_operation = False
        has_dict = False

        def visit_node(node):
            nonlocal for_loop_count, has_sorting_or_bisect, has_log_operation, has_dict
            if isinstance(node, ast.For):
                for_loop_count += 1
            # elif isinstance(node, ast.Dict):
            #     has_dict = True
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and node.func.attr == 'sort') or \
                   (isinstance(node.func, ast.Name) and node.func.id in {'bisect_left', 'bisect_right'}):
                    has_sorting_or_bisect = True
                elif isinstance(node.func, ast.Name) and node.func.id in {'sort', 'sorted'}:
                    has_sorting_or_bisect = True
                elif isinstance(node.func, ast.Name) and node.func.id in {'pow'}:
                    has_log_operation = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr in {'bit_length'}:
                    has_log_operation = True
                # elif isinstance(node.func, ast.Attribute) and node.func.attr in {'get', 'keys'}:
                #     has_dict = True
            for child in ast.iter_child_nodes(node):
                visit_node(child)

        visit_node(self.tree)

        if has_sorting_or_bisect:
            return 'nlogn'
        # elif has_dict and for_loop_count:
        #     return 'nlogn'
        # elif has_dict:
        #     return 'logn'
        elif has_log_operation:
            return 'logn'
        return ''
    
    # def find_built_in_func(self):
    #     functions = []
    #     for node in ast.walk(self.tree):
    #         if isinstance(node, ast.Call):
    #             if isinstance(node.func, ast.Name) and node.func.id in {'set', 'list'}:
    #                 functions.append(node.func.id)
    #             elif isinstance(node.func, ast.Attribute) and node.func.attr in {'count'}:
    #                 functions.append(node.func.attr)
    #     return functions

    def detect_time_complexity_for_built_in_func(self):
        has_linear_tc = False
        has_quadratic_tc = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {'Counter'}:
                    has_linear_tc = True
                if isinstance(node.func, ast.Attribute) and node.func.attr in {'count'}:
                    has_linear_tc = True
        if has_linear_tc:
            return 'n'
        return ''

    def calculate_time_complexity(self):
        loop_counts = self.extract_loop_counts()
        time_complexity_parts = self.calculate_function_complexity(loop_counts)

        total_complexity = " + ".join(time_complexity_parts) if time_complexity_parts else ""

        logn_nlogn_complexity = self.detect_time_complexity_for_logn_nlogn()
        if logn_nlogn_complexity:
            total_complexity += " + " + logn_nlogn_complexity if total_complexity else logn_nlogn_complexity

        built_in_complexity = self.detect_time_complexity_for_built_in_func()
        if built_in_complexity:
            total_complexity += " + " + built_in_complexity if total_complexity else built_in_complexity
        
        # print("Total Complexity: ", total_complexity)

        return total_complexity
    
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
                complexity_label = 2 if 'nl' not in part else 4
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


def process_code(code):
    calculator = TimeComplexityCalculator(code)
    if "def " in code:
        time_complexity = calculator.calculate_time_complexity_with_function_calls()
    else:
        time_complexity = calculator.calculate_time_complexity()
    return calculator.classify_time_complexity(time_complexity)


def save_result(src_csv_path, dest_csv_path):
    y_true = []
    y_pred = []
    with open(dest_csv_path, mode='w', newline='', encoding='utf-8') as dest_file:
        writer = csv.writer(dest_file)
        writer.writerow(['src', 'ground_truth', 'predicted_tc'])
        
        with open(src_csv_path, newline='', encoding='utf-8') as src_file:
            reader = csv.DictReader(src_file)
            for row in reader:
                src_code = row['src']
                ground_truth_label = row['complexity']
                predicted_label = process_code(src_code)
                writer.writerow([src_code, ground_truth_label, predicted_label])
                y_true.append(ground_truth_label)
                y_pred.append(predicted_label)

    y_true = [int(i) for i in y_true]
    accuracy = accuracy_score(y_true, y_pred)

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    classes = np.unique(np.concatenate((y_true, y_pred)))
    classes = list(map(int, classes))
    print(f"Classes: {classes}")
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
    plt.title('Confusion Matrix')
    output_file = 'confusion_matrix_python2.png'
    plt.savefig(output_file, bbox_inches='tight')

    print(f"Accuracy: {accuracy}")
    print(f"Micro-F1 Score: {micro_f1}")
    print(f"Macro-F1 Score: {macro_f1}")
    print("Class-wise Accuracy:")
    for cls, acc in classwise_accuracy.items():
        print(f"{cls}: {acc}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Calculate and save code complexity.")
    # parser.add_argument("--src_csv_path", default="./int_label/codecomplex_python/train.csv", help="Path to the source CSV file")
    # parser.add_argument("--dest_csv_path", default="./result.csv", help="Path to the destination CSV file to save results")
    # args = parser.parse_args()

    # save_result(args.src_csv_path, args.dest_csv_path)


    # hsan: when debugging with no saving csv, you can use like this.
    source_code = """
n,r=list(map(int,input().split()))
x=list(map(int,input().split()))
y=[r]*n
for i in range(1,n):
    for j in range(i):
        d=abs(x[i]-x[j])
        if d<=2*r:
            y[i]=max(y[i],y[j]+(4*r*r-d*d)**(0.5))
print(*y)
    """
    print(process_code(source_code))
