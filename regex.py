import re

def detect_and_save_double_for_loops(batch_data, save_file):
    double_for_loops = []

    pattern = r'for\s+\w+\s+in\s+\w+:\s*\n\s+for\s+\w+\s+in\s+\w+:'

    for data in batch_data:
        matches = re.findall(pattern, data)
        input(matches)      # hsan: for debugging
        for match in matches:
            double_for_loops.append(match)


    with open(save_file, 'w') as f:
        for loop in double_for_loops:
            f.write(loop + '\n')

# Example usage:
batch_data = [
    """
    for i in range(10):
        for j in range(5):
            print(i, j)
    """,
    """
    for x in range(5):
        for y in range(3):
            print(x, y)
    """,
    """
    for a in range(3):
        for b in range(2):
            print(a, b)
    """
]

save_file = 'double_for_loops.txt'

detect_and_save_double_for_loops(batch_data, save_file)

