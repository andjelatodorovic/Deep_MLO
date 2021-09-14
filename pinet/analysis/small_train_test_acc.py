import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('small_train.log')

groups = df.groupby(['dataset_name', 'train_size', 'model_params', 'split'])

best = df.iloc[groups.idxmin()['train_loss']]

groups_best = best.groupby(['dataset_name', 'train_size', 'model_params'])['test_acc']

means = groups_best.mean().reset_index().groupby('dataset_name')
stds = groups_best.std().reset_index().groupby('dataset_name')

for group_key in means.groups:
    fig = plt.figure()
    plt.title(str(group_key))
    _means = means.get_group(group_key)
    _stds = stds.get_group(group_key)

    # x_pos = np.arange(len(_means))
    plt.errorbar(_means['train_size'], _means['test_acc'], yerr=_stds['test_acc'])
    plt.xlabel('No. of training examples')
    plt.ylabel('Mean test accuracy (10 trials)')

    plt.show()
