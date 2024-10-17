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
from collections import defaultdict


class JavaTimeComplexityCalculator:
    def __init__(self, code):
        self.code = code
        try:
            self.tree = javalang.parse.parse(code)
        except javalang.parser.JavaSyntaxError as e:
            self.tree = None  # 구문 분석에 실패하면 트리를 None으로 설정
        self.function_complexities = {}

    def analyze_java_code(self):
        complexities = defaultdict(str)

        if self.tree is None:
            return complexities

        for _, node in self.tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                method_name = node.name
                method_body = node
                complexities[method_name] = self.extract_methods(method_body)
    
        print(complexities)
        return complexities

    def extract_loop_counts(self, method_body):
        loop_counts = []
        
        def visit_node(node):
            if isinstance(node, javalang.tree.ForStatement) or isinstance(node, javalang.tree.WhileStatement):
                loop_counts.append(node)
            for child in node.children:
                if isinstance(child, (list, tuple)):
                    for c in child:
                        if isinstance(c, javalang.ast.Node):
                            visit_node(c)
                elif isinstance(child, javalang.ast.Node):
                    visit_node(child)

        visit_node(method_body)
                
        return loop_counts

    def extract_methods(self, method_body):
        loop_counts = self.extract_loop_counts(method_body)
        time_complexity_parts = self.calculate_method_complexity(loop_counts)
        total_complexity = " + ".join(time_complexity_parts) if time_complexity_parts else ""
        return total_complexity

    def calculate_method_complexity(self, loop_counts):
        time_complexity_part = []
        nested_complexities = []

        for counts in loop_counts:
            depth = self.get_loop_depth(counts)
            complexity = self.get_complexity_of_loop(counts)
            if depth == 1:
                time_complexity_part.append(complexity)
            else:
                nested_complexities.append((depth, complexity, counts))

        nested_complexities.sort(reverse=True, key=lambda x: x[0])

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
            if isinstance(parent, (javalang.tree.ForStatement, javalang.tree.WhileStatement)):
                depth += 1
            current_node = parent
        return depth

    def get_parent_node(self, node):
        for path, parent in self.tree:
            for child in parent.children:
                if isinstance(child, (list, tuple)):
                    for c in child:
                        if c == node:
                            return parent
                elif child == node:
                    return parent
        return None

    def calculate_time_complexity(self):
        time_complexity_parts = []
        complexities = self.analyze_java_code()
        for method, complexity in complexities.items():
            time_complexity_parts.append(complexity)
        total_complexity = " + ".join(time_complexity_parts)

        return total_complexity

    def get_complexity_of_loop(self, loop_node):
        if isinstance(loop_node, javalang.tree.ForStatement):
            if self.detect_log_for_loop(loop_node):
                return 'logn'
            return 'n'
        elif isinstance(loop_node, javalang.tree.WhileStatement):
            if self.detect_log_while_loop(loop_node):
                return 'logn'
            return 'n'
        
        return '1'
    
    def detect_log_for_loop(self, node):
        condition = self.find_loop_condition(node)
        try:
            op = condition.update[0].postfix_operators[0]
        except:
            try:
                op = condition.update[0].type
            except:
                return False
        if op in ('*=', '/='):
            return True
        return False

    def detect_log_while_loop(self, node):
        condition = self.find_loop_condition(node)
        return False

    def find_loop_condition(self, node):
        for child in node.children:
            if isinstance(child, javalang.tree.ForControl):
                return child
            elif isinstance(child, javalang.tree.Expression):
                return child
            
        return None

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
    calculator = JavaTimeComplexityCalculator(code)
    time_complexity = calculator.calculate_time_complexity()
    return calculator.classify_time_complexity(time_complexity)

def save_result(src_csv_path, dest_csv_path):
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
    output_file = 'confusion_matrix_java.png'
    plt.savefig(output_file, bbox_inches='tight')

    print(f"Accuracy: {accuracy}")
    print(f"Micro-F1 Score: {micro_f1}")
    print(f"Macro-F1 Score: {macro_f1}")
    print("Class-wise Accuracy:")
    for cls, acc in classwise_accuracy.items():
        print(f"{cls}: {acc}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Calculate and save code complexity.")
    # parser.add_argument("--src_csv_path", default="./int_label/codecomplex_java/train.csv", help="Path to the source CSV file")
    # parser.add_argument("--dest_csv_path", default="./result_java.csv", help="Path to the destination CSV file to save results")
    # args = parser.parse_args()

    # save_result(args.src_csv_path, args.dest_csv_path)
    
    source_code = """
import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.Reader;
import java.io.InputStreamReader;
import java.io.InputStream;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        MyInput in = new MyInput(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskB solver = new TaskB();
        solver.solve(1, in, out);
        out.close();
    }

    static class TaskB {
        int n;
        MyInput in;
        PrintWriter out;

        public void solve(int testNumber, MyInput in, PrintWriter out) {
            this.in = in;
            this.out = out;

            n = in.nextInt();

            if (n / 2 % 2 == 1) {
                answer(-1);
                return;
            }

            int low = 0, high = n / 2;
            int diff = query(low + n / 2) - query(low);
            while (diff != 0) {
                int mid = (low + high) / 2;
                int d = query(mid + n / 2) - query(mid);
                if (d == 0 || diff > 0 == d > 0) {
                    diff = d;
                    low = mid;
                } else {
                    high = mid;
                }
            }
            answer(low);
        }

        int query(int i) {
            out.println("? " + (i % n + 1));
            out.flush();
            return in.nextInt();
        }

        void answer(int i) {
            out.println("! " + (i < 0 ? i : (i % n + 1)));
        }

    }

    static class MyInput {
        private final BufferedReader in;
        private static int pos;
        private static int readLen;
        private static final char[] buffer = new char[1024 * 8];
        private static char[] str = new char[500 * 8 * 2];
        private static boolean[] isDigit = new boolean[256];
        private static boolean[] isSpace = new boolean[256];
        private static boolean[] isLineSep = new boolean[256];

        static {
            for (int i = 0; i < 10; i++) {
                isDigit['0' + i] = true;
            }
            isDigit['-'] = true;
            isSpace[' '] = isSpace['\r'] = isSpace['\n'] = isSpace['\t'] = true;
            isLineSep['\r'] = isLineSep['\n'] = true;
        }

        public MyInput(InputStream is) {
            in = new BufferedReader(new InputStreamReader(is));
        }

        public int read() {
            if (pos >= readLen) {
                pos = 0;
                try {
                    readLen = in.read(buffer);
                } catch (IOException e) {
                    throw new RuntimeException();
                }
                if (readLen <= 0) {
                    throw new MyInput.EndOfFileRuntimeException();
                }
            }
            return buffer[pos++];
        }

        public int nextInt() {
            int len = 0;
            str[len++] = nextChar();
            len = reads(len, isSpace);
            int i = 0;
            int ret = 0;
            if (str[0] == '-') {
                i = 1;
            }
            for (; i < len; i++) ret = ret * 10 + str[i] - '0';
            if (str[0] == '-') {
                ret = -ret;
            }
            return ret;
        }

        public char nextChar() {
            while (true) {
                final int c = read();
                if (!isSpace[c]) {
                    return (char) c;
                }
            }
        }

        int reads(int len, boolean[] accept) {
            try {
                while (true) {
                    final int c = read();
                    if (accept[c]) {
                        break;
                    }
                    if (str.length == len) {
                        char[] rep = new char[str.length * 3 / 2];
                        System.arraycopy(str, 0, rep, 0, str.length);
                        str = rep;
                    }
                    str[len++] = (char) c;
                }
            } catch (MyInput.EndOfFileRuntimeException e) {
            }
            return len;
        }

        static class EndOfFileRuntimeException extends RuntimeException {
        }

    }
}
    """
    print(process_code(source_code))