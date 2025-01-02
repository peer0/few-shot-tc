import javalang
from collections import defaultdict


class JavaComplexityCalculator:
    def __init__(self, code):
        self.code = code
        try:
            self.tree = javalang.parse.parse(code)
        except javalang.parser.JavaSyntaxError as e:
            self.tree = None
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
