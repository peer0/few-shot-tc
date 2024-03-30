import re
import ast
import csv
import argparse

class TimeComplexityCalculator:
    def __init__(self, code):
        self.code = code

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

        if time_complexity_parts:
            total_complexity = " + ".join(time_complexity_parts)

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
    # 주어진 소스코드를 AST (Abstract Syntax Tree)로 파싱
        tree = ast.parse(self.code)

        # for 루프의 개수를 세는 변수
        for_loop_count = 0

        # 배열 관련 작업 여부를 확인하는 변수
        has_array_operation = False

        # AST를 순회하면서 for 루프의 개수와 배열 관련 작업 여부를 확인하는 함수
        def count_for_loops_and_array_operations(node):
            nonlocal for_loop_count, has_array_operation
            if isinstance(node, ast.For):
                for_loop_count += 1
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == 'arr':
                    has_array_operation = True
            for child in ast.iter_child_nodes(node):
                count_for_loops_and_array_operations(child)

        # AST 순회 시작
        count_for_loops_and_array_operations(tree)

        # for 루프가 1개 이상이면서 배열 관련 작업이 있으면 O(n log n) 시간 복잡도로 간주
        if for_loop_count >= 1 and has_array_operation:
            return 'nlogn'

        # for 루프가 1개 이상이면서 배열 관련 작업이 없고, 정렬 또는 이진 탐색이 있으면 O(log n) 시간 복잡도로 간주
        elif for_loop_count >= 1:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and node.func.attr == 'sort':
                        return 'logn'
                    elif isinstance(node.func, ast.Name) and (node.func.id == 'bisect_left' or node.func.id == 'bisect_right'):
                        return 'logn'

        return 'np'

    def classify_time_complexity(self, time_complexity):
        # '+' 기호로 복잡도 표현식을 분리
        complexity_parts = time_complexity.split(' + ')
        max_complexity_label = 1

        for part in complexity_parts:
            # '*' 기호로 각 부분을 분리
            factors = part.split(' * ')
            char_count = 0

            for f in factors:
                # 알파벳 문자가 포함되어 있거나 괄호를 포함하면, 변수 또는 함수 호출로 간주
                if any(char.isalpha() for char in f):
                    char_count += 1
            
            # logn == 2, nlogn == 4
            if 'log' in part:
                complexity_label = 2 if 'n' not in part else 4  
            # 복잡도 판별
            elif char_count == 0:
                complexity_label = 1  # constant == 1
            elif char_count == 1:
                complexity_label = 3  # linear == 3
            elif char_count == 2:
                complexity_label = 5  # quadratic == 5
            elif char_count == 3:
                complexity_label = 6  # cubic == 6
            else:
                complexity_label = 7  # np == 7

            # 최대 복잡도 레이블 갱신
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save code complexity.")
    parser.add_argument("--src_csv_path", default="D:\\workspace\\jointmatch\\data\\int_label\\codecomplex_python\\test.csv", help="Path to the source CSV file")
    parser.add_argument("--dest_csv_path", default="D:\\workspace\\jointmatch\\data\\result.csv", help="Path to the destination CSV file to save results")

    args = parser.parse_args()

    save_result(args.src_csv_path, args.dest_csv_path)



