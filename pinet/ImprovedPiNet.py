import json
import math
import os.path
import csv
import time
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import csv
import sys
import time
import argparse
import builtins
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from sklearn.model_selection import GridSearchCV
import pickle
from model.PiNet import PiNet
import os
import os.path as osp
import glob
torch.cuda.empty_cache()
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from typing import List
import wandb
from pathlib import Path
#from torchsampler import ImbalancedDatasetSampler


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}

        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        l = list(dataset.data.y)
        labels = []
        for i in range(len(l)):
            ind = torch.IntTensor.item(l[i])
            labels.append(ind)
            #print(labels)
        #return dataset.data.y[idx]
        return labels[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class TUDataset(InMemoryDataset):

    def __init__(self, root: str, name: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = 'DEEPMLO'
        # self.raw_dir = os.path.join(Path().resolve().parent, 'DEEPMLO')
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
                                           'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels', torch.long)
        if node_labels.dim() == 1:
            node_labels = node_labels.unsqueeze(-1)
        node_labels = node_labels - node_labels.min(dim=0)[0]
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
    x = cat([node_attributes, node_labels])

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)

    return data, slices


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    print(path)
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def get_labels():
    data, slices = read_tu_data(os.path.join(Path().resolve().parent, 'DEEPMLO'), 'DEEPMLO')
    return data.y

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        data.num_nodes = torch.bincount(batch).tolist()
        slices['num_nodes'] = torch.arange(len(data.num_nodes) + 1)
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def info(*msg):
    old_print(*msg, file=sys.stderr)


old_print = builtins.print
builtins.print = info

def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions).argmax(axis=1)
    print(metrics.confusion_matrix(labels, predictions).ravel())
    TN, FP, FN, TP = metrics.confusion_matrix(labels, predictions).ravel()

    # print(TP, FP, FN)


    if(TP+FP == 0):
        precision_1 = 0
    else:
        precision_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)

    if (TN + FN == 0):
        precision_0 = 0
    else:
        precision_0 = TN / (TN + FN)

    recall_0 = TN / (TN + FP)
    if (precision_0 + recall_0 == 0):
        f1_0 = 0
    else:
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    if (precision_1 + recall_1 == 0):
        f1_1 = 0
    else:
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

    return accuracy_score(labels, predictions), precision_1, recall_1,precision_0, recall_0, f1_0, f1_1


def get_splits(train_size):
    s = StratifiedShuffleSplit(n_splits=10, train_size=train_size).split(np.zeros([len(dataset), 1]), dataset.data.y)
    print(s)
    return s

if __name__ == '__main__':
    experiment_count = 1


    log_file = open('small_train_GCN.log', 'a')
    writer = csv.writer(log_file)
    datasets = ["DEEPMLO"]

    models = [
        {
            'class': PiNet,
            'params': {
                'message_passing': 'GCN',
                'GCN_improved': False,
                'dims': [32, 64],
            },
        },

    ]
    num_folds = 10
    train_sizes = [0.8]
    CLASS_0_WEIGHT = -0.0217
    CLASS_1_WEIGHT = -0.0584
    #train_sizes = []

    device = "cpu"
    print(f'exp: {experiment_count}, running on {device}')
    writer.writerow([experiment_count,
                     'dataset_name',
                     'model',
                     'model_params',
                     'train_size',
                     'split',
                     'epoch',
                     'time_for_epoch(ms)',
                     'train_loss',
                     'train_acc', 'test_acc',
                     'train_conf', 'test_conf'])
    log_file.flush()

    for dataset_name in datasets:
        for model_dict in models:
            for train_size in train_sizes:


                dataset = TUDataset(root=os.path.join(Path().resolve().parent, 'DEEPMLO'), name=dataset_name).shuffle()

                Directory = Path(Path.cwd()).parent
                file = os.path.join(Directory, 'train_test_indices.txt')
                with open(file, "rb") as fp:   # Unpickling
                    train_test_list = pickle.load(fp)
                #print(list)

                for split in range(num_folds):

                    for split, (all_train_idx, test_idx) in enumerate(train_test_list):

                    #index_class_1 = list(labels_df[labels_df.label == 1].index)
                    #index_class_0 = list(labels_df[labels_df.label == 0].index)

                    #train_indicies_class_1, test_indicies_class_1 = shuffle_and_pick(index_class_1, 14)
                    #train_indicies_class_0, test_indicies_class_0 = shuffle_and_pick(index_class_0, 47)

                    #train_indicies = train_indicies_class_1 + train_indicies_class_0
                    #test_indicies = test_indicies_class_1 + test_indicies_class_0

                    #train_dataset = [patients_dataset[i] for i in train_indicies]
                    #test_dataset = [patients_dataset[i] for i in test_indicies]

                    #train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
                    #test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)



                        train_idx = torch.tensor(all_train_idx, dtype=torch.long)
                        test_idx = torch.tensor(test_idx, dtype=torch.long)
                        train_loader = DataLoader(dataset[train_idx], batch_size=len(train_idx),
                                                  sampler=ImbalancedDatasetSampler(dataset[train_idx]))
                        test_loader = DataLoader(dataset[test_idx], batch_size=len(test_idx))

                        run = wandb.init(reinit=True, project="PINET_Balanced")
                        print(dataset_name, model_dict['class'].__name__)

                        model = model_dict['class'](num_feats=dataset.num_features,
                                                    num_classes=dataset.num_classes,
                                                    **model_dict['params']).to(device)

                        # convert idx to torch tensors
                        #train_idx = torch.tensor(all_train_idx, dtype=torch.long)
                        #test_idx = torch.tensor(test_idx, dtype=torch.long)

                        #train_loader = DataLoader(dataset[train_idx], batch_size=len(train_idx))
                        #test_loader = DataLoader(dataset[test_idx], batch_size=len(test_idx))


                        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
                        WEIGHTS = torch.FloatTensor([CLASS_0_WEIGHT, CLASS_1_WEIGHT])
                        crit = CrossEntropyLoss(weight = WEIGHTS)



                        for epoch in range(200):
                            start = time.time()
                            train_loss = train()
                            time_for_epoch = (time.time() - start) * 1e3
                            train_acc, precision_1_train, recall_1_train,precision_0_train, recall_0_train, f1_0_train, f1_1_train = evaluate(train_loader)
                            test_acc, precision_1, recall_1,precision_0, recall_0, f1_0, f1_1 = evaluate(test_loader)

                            writer.writerow([dataset_name,
                                                model_dict["class"].__name__,
                                                model_dict['params'],
                                                train_size,
                                                split,
                                                epoch,
                                                time_for_epoch,
                                                train_loss,
                                                train_acc, test_acc,
                                                precision_1, recall_1,
                                                precision_0, recall_0, f1_0, f1_1])

                            log_file.flush()
                            wandb.log({"loss_PiNet_GCN": np.mean(train_loss),"Train accuracy_PiNet_GCN": train_acc, "Test acuuracy_PiNet_GCN":test_acc, 'precision_1_PiNet_GCN':precision_1,'recall_1_PiNet_GCN':recall_1,'precision_0_PiNet_GCN': precision_0,'recall_0_PiNet_GCN': recall_0, 'f1_0_PiNet_GCN':f1_0, 'f1_1_PiNet_GCN':f1_1})
                        run.finish()