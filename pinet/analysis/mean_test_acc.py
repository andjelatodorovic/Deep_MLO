import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('pinet_hypers.log')
df.drop(df[df['epoch'] < 200].index, inplace=True)
df.reset_index(inplace=True)

groups = df.groupby(['dataset_name', 'model_params', 'split'])

best = df.iloc[groups.idxmax()['test_acc']]

test_accs = best.groupby(['dataset_name', 'model_params'])['test_acc']

means = test_accs.mean().reset_index().groupby('dataset_name')
stds = test_accs.std().reset_index().groupby('dataset_name')

fig, nested_axes = plt.subplots(4, 2, sharex='all')
axes = [ax for row in nested_axes for ax in row]

fig.set_size_inches(8.27, 11.69)    # A4

x_pos = np.arange(len(df['model_params'].unique()))
xlabels = df['model_params'].apply(lambda line: line.replace('\',', '\'\n')).unique()
plt.xticks(x_pos, xlabels, rotation=90)

for ax, group_key in zip(axes, means.groups):
    ax.set_title(str(group_key))
    _means = means.get_group(group_key)
    _stds = stds.get_group(group_key)

    ax.bar(np.arange(len(_means)), _means['test_acc'], yerr=_stds['test_acc'])


plt.tight_layout()
# plt.savefig('mean_test_acc_allow_200_for_convergence.pdf')
plt.show()
