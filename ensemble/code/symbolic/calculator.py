import re
import ast
import numpy as np
import os
from os import walk
import sys
from time import sleep
import javalang
from javalang import parse



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

def analyze_loops(tree):
    loop_countinfo = []
    loop_count = "1 "
    for _, node in tree:
        if analyze_method(node)=="1 ": continue
        loop_count += f" + {analyze_method(node)} "
    return loop_count

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

def find_loop_condition(loop_node):
    # Traverse the children of the loop node to find the condition expression
    for child in loop_node.children:
        if isinstance(child, javalang.tree.ForControl):
            return child
        elif isinstance(child, javalang.tree.Expression):
            return child

    return None  # Condition not found


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
    
def extract_loop_counts_and_function_calls(func_code):    # hsan: this is for python
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

def extract_loop_counts(code):
    if not isinstance(code, str):
        raise TypeError("Expected a string for 'code' parameter")
    loop_counts = []
    lines = code.split('\n')
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

class CodeComplexityCalculator:
    def __init__(self, code, language):
        self.code = code
        self.language = language.lower()
        self.success_method = 0
        self.error_method = 0

    def extract_functions(self):
        if self.language == 'java':
            functions = []
            invocations = []
            loop_counts = []
            function_calls = []
            for_num = 0
            while_num = 0
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
        
        elif self.language == 'python':
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
        
        else:
            raise NotImplementedError
        
    def log_detector(self):
        # tree = ast.parse(self.code)
        analyzer = LoopAndRecursiveAnalysis(self.code, self.language)
        # analyzer.visit(tree)
        results, contains_sort = analyzer.analyze()
        if analyzer.contains_sort:
            return 'nlogn'
        elif analyzer.results:
            return 'logn'
        else:
            return ""


    def calculate_time_complexity_with_function_calls(self):
        if self.language == 'java':
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
        
            total_complexity = "".join(" + ".join(function_loop_counts).split())
        
        elif self.language == 'python':
            functions = self.extract_functions()
            function_call_counts = {func: 0 for func in functions}
            function_loop_counts = {}
            function_call_names = {}

            for func_code in functions:
                match = re.search(r'def (\w+)\(', func_code)
                if not match:
                    continue
                function_name = match.group(1)
                loop_counts, function_calls = extract_loop_counts_and_function_calls(func_code)
                function_loop_counts[function_name] = loop_counts
                function_call_names[function_name] = function_calls

            for func_name, calls in function_call_names.items():
                for call in calls:
                    if call in function_call_counts:
                        function_call_counts[call] += 1  # 호출 횟수 갱신

            time_complexity_parts = []
            time_complexity_part = []
            total_complexity = ""

            for func_name, loop_counts in function_loop_counts.items():
                time_complexity_part = []
                for count in loop_counts:
                    time_complexity_part.append(" * ".join(count))
                for call_name in function_call_names.get(func_name, []):
                    if call_name in function_call_counts:
                        time_complexity_part.append(f"{function_call_counts[call_name]}")
                time_complexity_parts.append("(" + " * ".join(time_complexity_part) + ")")

            # hsan: check for logn and nlogn complexities
            logn_nlogn_complexity = self.log_detector()
            if logn_nlogn_complexity:
                time_complexity_part.append(logn_nlogn_complexity)

            total_complexity = " + ".join(time_complexity_parts) if time_complexity_parts else ""

        else:
            raise NotImplementedError

        return total_complexity
    
    def calculate_time_complexity(self):    # hsan: this is for python, input code without function
        loop_counts = extract_loop_counts(self.code)
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

    def classify_time_complexity(self, time_complexity):
        complexity_parts = time_complexity.split('+')
        max_complexity_label = -1
        complexity_label = -1
        valid_complexity = False  # Initialize valid_complexity
        
        for part in complexity_parts:
            factors = part.split('*')
            char_count = 0

            for f in factors:
                if any(char.isalpha() for char in f):
                    char_count += 1

            if self.language == 'java':
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

            elif self.language == 'python':
                # logn == 2, nlogn == 4
                if 'log' in part:
                    complexity_label = 2 if 'n' not in part else 4  
                elif '>>' in part:
                    for f in part.split('>>')[1]:
                        if any(char.isalpha() for char in f):
                            if char_count == 0:
                                complexity_label = 2
                            elif char_count == 1:
                                complexity_label = 4  
                # 복잡도 판별
                elif char_count == 0:
                    complexity_label = 1  # constant == 1
                elif char_count == 1:
                    complexity_label = 3  # linear == 3
                elif char_count == 2:
                    complexity_label = 5  # quadratic == 5
                elif char_count == 3:
                    complexity_label = 6  # cubic == 6
                elif "<<" in part:
                    for f in part.split('<<')[1]:
                        if any(char.isalpha() for char in f):
                            complexity_label = 7  # np
                else:
                    complexity_label = -100
            
            
            if complexity_label > 0:
                valid_complexity = True
            max_complexity_label = max(max_complexity_label, complexity_label)

        if not valid_complexity:
            return -100
            #raise ValueError("Unable to determine time complexity for the given input!")


        return int(max_complexity_label)
    

class LoopAndRecursiveAnalysis(ast.NodeVisitor):
    def __init__(self, code, language):
        self.code = code
        self.language = language.lower()
        self.results = []
        self.contains_sort = False

    def analyze(self):
        if self.language == 'python':
            tree = ast.parse(self.code)
            self.visit(tree)
        elif self.language == 'java':
            tree = javalang.parse.parse(self.code)
            self.visit_java(tree)
        else:
            raise NotImplementedError(f"We do not support {self.language}, sorry :( )")
        return self.results, self.contains_sort


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

    def visit_java(self, tree):
        for path, node in tree:
            if isinstance(node, javalang.tree.WhileStatement):
                self.visit_java_while(node)
            elif isinstance(node, javalang.tree.MethodInvocation):
                self.visit_java_call(node)

    def visit_java_while(self, node):
        for statement in node.body:
            if isinstance(statement, javalang.tree.StatementExpression):
                expr = statement.expression
                if isinstance(expr, javalang.tree.Assignment):
                    if isinstance(expr.expression, javalang.tree.BinaryOperation) and isinstance(expr.expression.operator, str):
                        if expr.expression.operator == '/':
                            if isinstance(expr.expression.operandr, javalang.tree.Literal) and expr.expression.operandr.value == '2':
                                self.results.append(('While loop', node.position.line, expr.expression.operandl.member))

    def visit_java_call(self, node):
        if isinstance(node, javalang.tree.MethodInvocation):
            if node.member in ['sort', 'sorted']:
                self.contains_sort = True
            for argument in node.arguments:
                if isinstance(argument, javalang.tree.BinaryOperation):
                    if argument.operator == '/':
                        if isinstance(argument.operandr, javalang.tree.Literal) and argument.operandr.value == '2':
                            self.results.append(('Recursive call', node.position.line, node.member))
