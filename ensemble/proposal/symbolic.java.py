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


def get_method_start_end(method_node,tree):
    startpos  = None
    endpos    = None
    startline = None
    endline   = None
    method_name = None
    for path, node in tree:
        if startpos is not None and method_node not in path:
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            if isinstance(node, javalang.tree.ForStatement):
                method_name = "DUMMY"
            elif isinstance(node, javalang.tree.WhileStatement):
                method_name = "DUMMY"
            else:
                method_name = node.name if node.name is not None else "DUMMY"
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    return startpos, endpos, startline, endline, method_name

def find_end_of_function(code, start_line):
    lines = code.split('\n')
    function_code = []
    stack = []
    function_started = False
    multi_comment = False

    # Iterate over the lines starting from the start line number
    for line_number in range(start_line - 1, len(lines)):
        line = lines[line_number]
        count_brace = 0
        
        # Check for comments and skip them
        if line.startswith('//'):
            continue
        elif line.startswith('/*'):
            multi_comment = True
            continue
        elif multi_comment:
            if line.endswith('*/'):
                multi_comment = False
                continue
        if function_started:
            if not stack:
                break

        # Iterate over characters in the line
        for char in line:
            if char == '{':
                function_started = True
                stack.append('{')
            elif char == '}':
                if stack:
                    count_brace+=1
                    stack.pop()
                else:
                    # If the stack is empty, it indicates the end of the function
                    function_code.append(line[:line.find('}')+1])
                    function_code = '\n'.join(function_code)
                    return function_code
            # Append the line to the function code
        function_code.append(line)

    # If the stack is not empty after iterating through all lines, 
    # it means the function is not closed properly.
    if stack:
        raise SyntaxError("Function is not properly closed.")
    
    function_code = '\n'.join(function_code)
    return function_code

def analyze_method(node):
    loop_count = "1 "
    if isinstance(node, javalang.tree.ForStatement):
        loop_type = 'for'
        loop_condition = find_loop_condition(node)
        loop_count = f"{count_iterations(loop_condition)} "
        if node.body:
            for _,statement in node.body:
                if analyze_method(statement) != "1 ":
                    loop_count += f"* ({analyze_method(statement)}) "
    elif isinstance(node, javalang.tree.WhileStatement):
        loop_type = 'while'
        loop_condition = find_loop_condition(node)
        loop_count = f"{count_iterations(loop_condition)} "
        if node.body:
            for _, statement in node.body:
                if analyze_method(statement) != "1 ":
                    loop_count += f"* ({analyze_method(statement)}) "
    return loop_count
    
def analyze_loops(tree):
    loop_countinfo = []
    loop_count = "1 "
    for _, node in tree:
        if analyze_method(node)=="1 ": continue
        loop_count += f" + {analyze_method(node)} "
    return loop_count


def find_loop_condition(loop_node):
    # Traverse the children of the loop node to find the condition expression
    for child in loop_node.children:
        if isinstance(child, javalang.tree.ForControl):
            return child
        elif isinstance(child, javalang.tree.Expression):
            return child

    return None  # Condition not found

def extract_expression(loop_node,node):
    if isinstance(node, javalang.tree.BinaryOperation):
        left_operand = extract_expression(loop_node,node.operandl)
        operator = node.operator
        right_operand = extract_expression(loop_node,node.operandr)
        if operator == '*':
            return f"{left_operand} {operator} {right_operand}"
        elif operator in ('&&', '||', '!=', '=='):
            # Binary logical operator
            return 1  # Cannot determine iterations for complex conditions
        elif operator in ('<', '>', '<=', '>='):
            if isinstance(left_operand, javalang.tree.Literal):
                start_value = int(left_operand.value)
            else:
                start_value = loop_node.init.declarators[0].initializer.value
            if isinstance(right_operand, javalang.tree.Literal):
                end_value = int(right_operand.value)
            else:
                end_value = right_operand
            if operator in ('>', '>='):
                start_value, end_value = end_value, start_value
            return end_value
        else:
            return f"{left_operand} {operator} {right_operand}"
    elif isinstance(node, javalang.tree.MemberReference):
        if node.qualifier:
            return f"{node.qualifier}.{node.member}"
        else:
            return node.member
    elif isinstance(node, javalang.tree.Literal):
        return node.value
    elif isinstance(node, javalang.tree.MethodInvocation):
        if node.qualifier:
            method_name = f"{node.qualifier}.{node.member}"
        else:
            method_name = node.member
        if len(node.arguments) == 0:
            return method_name + "()"
        else:
            arguments = ", ".join([extract_expression(loop_node,arg) for arg in node.arguments])
        return f"{node.member}({arguments})"
    elif isinstance(node, javalang.tree.StatementExpression) and isinstance(node.expression, javalang.tree.MethodInvocation):
        method_name = node.expression.member
        if len(node.expression.arguments) == 0:
            return method_name + "()"
        else:
            arguments = ", ".join([extract_expression(loop_node, arg) for arg in node.expression.arguments])
        return f"{node.expression.member}({arguments})"
    else:
        return 1

def count_iterations(loop_node):
    try:
        loop_condition = loop_node.condition
        loop_initializer = loop_node.init.declarators[0].initializer.value
    except:
        return 1
    if isinstance(loop_condition, javalang.tree.BinaryOperation):
        expression = extract_expression(loop_node,loop_condition)
        if expression == None:
            print(loop_node, loop_condition)
            input()
        return expression
    else:
        return 1
        # Assuming simple binary expressions for condition

def count_function_calls(function_name, code):
    class FunctionCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.call_count = 0

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                self.call_count += 1
            self.generic_visit(node)

    tree = ast.parse(code)
    visitor = FunctionCallVisitor()
    visitor.visit(tree)

    return visitor.call_count



class CodeComplexityCalculator:
    def __init__(self, code, language):
        self.code = code
        self.language = language.lower()
        self.success_method = 0
        self.error_method = 0

    def extract_functions(self):
        functions = []
        invocations = []
        loop_counts = []
        function_calls = []
        for_num = 0
        while_num = 0
        if self.language == 'java':
            try:
                tree = javalang.parse.parse(self.code)
            except Exception as e:
                return functions,invocations

            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                startpos, endpos, startline, endline, method_name = get_method_start_end(node,tree)
                method = find_end_of_function(self.code,startline)
                functions.append({"language":self.language, "name":method_name, "method":method})
            for path, node in tree.filter(javalang.tree.ForStatement):
                startpos, endpos, startline, endline, method_name = get_method_start_end(node,tree)
                method_name = "for{}".format(for_num)
                method = find_end_of_function(self.code,startline)
                for_num+=1
                invocations.append({"language":self.language, "name":method_name, "method": method, "node":node})
            for path, node in tree.filter(javalang.tree.WhileStatement):
                startpos, endpos, startline, endline, method_name = get_method_start_end(node,tree)
                method_name = "while{}".format(while_num)
                method = find_end_of_function(self.code,startline)
                while_num+=1
                invocations.append({"language":self.language, "name":method_name, "method": method, "node":node})
        return functions, invocations


    def calculate_time_complexity_with_function_calls(self):
        functions,invocations = self.extract_functions()
        function_call_counts = {func["name"]: 0 for func in functions}
        function_loop_counts = []
        function_call_names = {}
        function_name = "ERROR"

        for func_code in functions:
            function_language = func_code["language"]
            function_name = func_code["name"]
            function_method = func_code["method"]
            try:
                func_to_parse = javalang.tokenizer.tokenize(function_method)
                parser = javalang.parser.Parser(func_to_parse)
            except Exception as e:
                #print("*"*200)
                #print(function_method)
                #print("*"*200)
                #print(e)
                #print("*"*200)
                function_loop_counts.append('')
                self.error_method += 1
                continue
            try:
                partial_tree = parser.parse_member_declaration()
            except Exception as e:
                function_loop_counts.append('')
                self.error_method += 1
                continue
            loop_info = analyze_loops(partial_tree)
            function_loop_counts.append(loop_info)
            self.success_method += 1
        for invoc in invocations:
            loop_info = analyze_loops(invoc["node"])
            function_loop_counts.append(loop_info)
            
            #loop_counts, function_calls = self.extract_loop_counts_and_function_calls(function_method)
        total_complexity = "".join(" + ".join(function_loop_counts).split())

        return total_complexity

    def classify_time_complexity(self, time_complexity):
        complexity_parts = time_complexity.split('+')
        max_complexity_label = -1

        for part in complexity_parts:
            factors = part.split('*')
            char_count = 0

            for f in factors:
                if any(char.isalpha() for char in f):
                    char_count += 1
            
            if '>>' in part:
                for f in part.split('>>')[1]:
                    if any(char.isalpha() for char in f):
                        if char_count == 0:
                            complexity_label = 2
                        elif char_count == 1:
                            complexity_label = 4  
            if char_count == 0:
                complexity_label = 1  # constant
            elif char_count == 1:
                complexity_label = 3  # linear
            elif char_count == 2:
                complexity_label = 5  # quadratic
            elif char_count == 3:
                complexity_label = 6  # cubic
            elif "<<" in part:
                for f in part.split('<<')[1]:
                    if any(char.isalpha() for char in f):
                        complexity_label = 7  # np
            else:
                complexity_label = -100

            max_complexity_label = max(max_complexity_label, complexity_label)

        return int(max_complexity_label)


def process_code(code, language, ground_truth):
    calculator = CodeComplexityCalculator(code, language)
    time_complexity = calculator.calculate_time_complexity_with_function_calls()
    prediction = calculator.classify_time_complexity(time_complexity)
    return time_complexity, prediction, calculator.error_method, calculator.success_method


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
                complexity_prediction, prediction, error_method, success_method = process_code(src_code, language, ground_truth_label)
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
    output_file = 'confusion_matrix.png'
    plt.savefig(output_file, bbox_inches='tight')  # Saves the plot as a PNG file

    print(f"Accuracy: {accuracy}")
    print(f"Micro-F1 Score: {micro_f1}")
    print(f"Macro-F1 Score: {macro_f1}")
    print("Class-wise Accuracy:")
    for cls, acc in classwise_accuracy.items():
        print(f"{cls}: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save code complexity.")
    parser.add_argument("--src_path", default="", help="Path to the source CSV file")
    parser.add_argument("--dest_path", default="", help="Path to the destination CSV file to save results")
    parser.add_argument("--language", default="python", help="Programming language of the source code (python/java)")

    args = parser.parse_args()

    save_result(args.src_path, args.dest_path, args.language)

