import ast

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

def analyze_code(source_code):
    tree = ast.parse(source_code)
    analyzer = LoopAndRecursiveAnalysis()
    analyzer.visit(tree)
    
    if analyzer.contains_sort:
        return 'nlogn'
    elif analyzer.results:
        return 'logn'
    else:
        return 'no matching complexity!'

# Example code
source_code = """
#!/usr/bin/python

x,k = map(int, input().strip().split())

MOD = 1000000007

if x > 0:
	r = (pow(2, k+1, MOD) * x - pow(2, k, MOD) + 1 + MOD * 10) % MOD
else:
	r = 0

print(r)
"""

# Run analysis
result = analyze_code(source_code)

# Print result
print("Complexity:", result)