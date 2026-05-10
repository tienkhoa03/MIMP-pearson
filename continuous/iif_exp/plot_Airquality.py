import os
import numpy as np
import matplotlib.pyplot as plt

from plot_helper import collect_dataset_metrics


def plot_grouped_bar(xlabels, data: dict, ylabel: str, title: str, outpath: str):
    labels = list(data.keys())
    n_groups = len(xlabels)
    n_labels = len(labels)
    x = np.arange(n_groups)
    width = 0.15

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure(figsize=(8, 4.5))
    for i, lbl in enumerate(labels):
        vals = [data[lbl][i] for i in range(n_groups)]
        plt.bar(x + (i - (n_labels - 1) / 2) * width, vals, width, label=lbl, color=colors[i % len(colors)])
    plt.xticks(x, xlabels)
    plt.xlabel('Ratio of stream (%)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    base = os.path.dirname(__file__)
    folder = os.path.join(base, 'Airquality')
    dataset = 'Airquality'
    streams = [0.001, 0.01, 0.1, 1.0]
    metrics = collect_dataset_metrics(folder, dataset, streams)

    mre_data = {k: [metrics[k][str(s)]['mre'] for s in streams] for k in metrics}
    time_data = {k: [metrics[k][str(s)]['time'] for s in streams] for k in metrics}

    out_mre = os.path.join(folder, 'Airquality_MRE.png')
    out_time = os.path.join(folder, 'Airquality_Time.png')

    plot_grouped_bar(['0.1', '1', '10', '100'], mre_data, 'MRE (%)', f'{dataset} - MRE (%)', out_mre)
    plot_grouped_bar(['0.1', '1', '10', '100'], time_data, 'Time (s)', f'{dataset} - Time (s)', out_time)

    print('Saved:', out_mre, out_time)


if __name__ == '__main__':
    main()
