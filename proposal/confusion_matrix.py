import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

#np.random.seed(0)  # For reproducibility
input_file = open(sys.argv[1]).readlines()
y_true = []
y_pred = []
labels = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'exponential', 'ERROR']

with open(sys.argv[1]) as inputfile:
    for line in inputfile:
        content = json.loads(line)
        if content['answer'] == 'np':
            y_true.append("exponential")
        else:
            y_true.append(content['answer'])
        if content['complexity'] not in labels:
            y_pred.append("ERROR")
        else:
            y_pred.append(content['complexity'])


#labels = sorted(set(y_true + y_pred))


# Generating the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
# plt.show()

output_file = '{}.confusion_matrix.png'.format(sys.argv[1])
plt.savefig(output_file, bbox_inches='tight')  # Saves the plot as a PNG file
plt.close()  # Closes the plot to prevent it from displaying in the notebook


# Calculate Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate Micro-F1 and Macro-F1 scores
micro_f1 = f1_score(y_true, y_pred, average='micro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Micro-F1 Score: {micro_f1}")
print(f"Macro-F1 Score: {macro_f1}")
