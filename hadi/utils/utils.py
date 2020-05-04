import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


def convert_time(time_in_secs):
    d = time_in_secs // 86400
    h = (time_in_secs - d * 86400) // 3600
    m = (time_in_secs - d * 86400 - h * 3600) // 60
    s = time_in_secs - d * 86400 - h * 3600 - m * 60

    print("\nd / hh:mm:ss   --->   %d / %d:%d:%d\n" % (d, h, m, s))


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, dtype='long'):
    if dtype == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))
    elif dtype == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor))


def view_input(x, nlp, conversion_dict=None):
    if conversion_dict is None:
        print(" ".join([nlp.i2w[t] for t in to_np(x)]))
    else:
        print([[nlp.i2w[t] for t in conversion_dict[obj]] for obj in to_np(x)])


def plot_results(stats, labels=None, normalize=True, figsize=(12, 6), legend_loc='lower right', savefig=None):
    if type(stats) is not list:
        stats = [stats]
    if type(labels) is not list:
        labels = [labels]

    assert len(stats) == len(labels), "stats and labels should have the same length"

    data_list, data_to_plot_list = [], []
    for stat in stats:
        nb_episodes = stat["nb_episodes"]
        max_step = stat["max_step"]

        data = pd.DataFrame(index=range(max_step))
        for t in range(nb_episodes):
            if normalize:
                scores_in_episode = (np.array(stat['scores']['episode_%d' % t]) /
                                     stat['max_score']['episode_%d' % t] * 100)
            else:
                scores_in_episode = np.array(stat['scores']['episode_%d' % t]).astype(float)

            df = pd.DataFrame(scores_in_episode, columns=['ep_%d' % t])
            data = pd.concat([data, df], axis=1)
        data_list.append(data)

        data_to_plot = pd.DataFrame(columns=['score', 'step'])
        for step in range(max_step):
            lst = list(zip(list(data.iloc[step]), [step] * max_step))
            df = pd.DataFrame(lst, columns=['score', 'step'])
            data_to_plot = pd.concat([data_to_plot, df], axis=0)
        data_to_plot_list.append(data_to_plot)

    plt.figure(figsize=figsize)
    for indx, df in enumerate(data_to_plot_list):
        sns.lineplot(x="step", y="score", data=df, label=labels[indx])
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Score %s" % "%", fontsize=20)
    plt.grid()
    if None not in labels:
        plt.legend(loc=legend_loc, fontsize=20)
    if savefig is not None:
        plt.savefig(savefig, facecolor="white")
    plt.show()

    return data_list, data_to_plot_list
