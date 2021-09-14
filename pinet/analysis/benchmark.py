import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('benchmark.log')

groups = df.groupby(['dataset_name', 'model_params', 'split'])
for i, group_key in enumerate(groups.groups):
    fig = plt.figure()
    group = groups.get_group(group_key)
    plt.plot(group['epoch'], group[['train_loss', 'train_acc', 'val_acc', 'test_acc']], '-')
    plt.title(str(group_key))
    plt.legend(['train_loss', 'train_acc', 'val_acc', 'test_acc'])
    plt.savefig(f'plots/{str(group_key)}.png')
    plt.close(fig)

