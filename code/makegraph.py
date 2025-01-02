import numpy as np
import matplotlib.pyplot as plt


# Define global font sizes using matplotlib.rcParams
plt.rcParams.update({'font.size': 18,  # Sets the base default font size
                     'axes.labelsize': 26,  # Font size for x and y labels
                     'axes.titlesize': 26,  # Font size for plot titles
                     'xtick.labelsize': 20,  # Font size for x-axis tick labels
                     'ytick.labelsize': 20,  # Font size for y-axis tick labels
                     'legend.fontsize': 14})  # Font size for legends

# Define models and datasets
models = ['CodeBERT', 'GraphCodeBERT', 'UniXcoder', 'CodeT5+', 'TCProF(CodeT5+)', 'TCProF(UniXcoder)']
categories = ['CodeComplex (Java)', 'CodeComplex (Python)', 'CorCoD']

# Example data for Accuracy and F1 scores
data_acc = np.array([
    [39.82, 66.39, 74.74],
    [46.46, 67.83, 72.98],
    [46.76, 68.44, 77.54],
    [55.11, 72.88, 75.79],
    [41.98, 59.29, 51.93],
    [48.70, 70.29, 63.16]
])

data_f1 = np.array([
    [37.35, 54.34, 76.64],
    [37.75, 54.13, 76.48],
    [38.76, 55.45, 81.69],
    [44.14, 56.39, 79.51],
    [33.60, 44.03, 49.60],
    [49.45, 53.17, 63.57]
])

colors = plt.get_cmap('Blues')(np.linspace(0.2, 1, len(models)))  # Assign colors from a colormap

# Function to create bar plots
def create_bar_plot(data, filename, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.15  # Width of bars

    for i, model in enumerate(models):
        # Calculate positions for bars
        positions = np.arange(len(categories)) * (len(models) * bar_width + 0.1) + i * bar_width
        ax.bar(positions, data[i, :], bar_width, color=colors[i], label=model, edgecolor='black')

    # Set labels and ticks
    ax.set_ylabel('Scores (%)')
    ax.set_xticks(np.arange(len(categories)) * (len(models) * bar_width + 0.1) + bar_width * (len(models)/2 - 0.5))
    ax.set_xticklabels(categories)
#    ax.set_ylim(0, 100)  # Ensure y-axis is scaled from 0 to 100
    ax.set_xlabel('Dataset')  # Label x-axis
    ax.legend(loc='upper left')
    #ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the plot with high resolution
    plt.close()  # Close the figure to free up memory

# Create and save each plot
create_bar_plot(data_acc, 'full_train_graph_acc.png', 'Accuracy Performance')
create_bar_plot(data_f1, 'full_train_graph_f1.png', 'F1 Performance')
