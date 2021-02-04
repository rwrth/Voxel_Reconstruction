import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import dgl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def from_binary(path):
    f = open(path, 'rb')
    objs = []
    while 1:
        try:
            o = pickle.load(f)
            objs.append(o)
        except EOFError:
            break
    return objs

def PlotOne(label):
    truth = label[1]
    la = np.reshape(truth,(20,20,20))
    #label_norm = np.divide(truth,np.max(truth))
    #label_norm = np.reshape(label_norm,(20,20,20))
    filled2 = (la > 0)
    x,y,z = np.mgrid[-2:2:21j, -2:2:21j, -2:2:21j]

    colors2 = np.empty((20,20,20,4), dtype=np.float32)
    colors2[:,:,:,:3] = [1,0,0]
    colors2[:,:,:,3] = filled2*0.5

    fig = plt.figure(figsize=(10,12))
    fig.suptitle('Voxels', fontsize=20)
    ax = fig.gca(projection='3d')
    voxels = ax.voxels(x,y,z, filled2, facecolors=colors2, edgecolors= "black", linewidth=0.5, label ="Truth")
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    fig.savefig("t_prep.pdf", format = "pdf")
    plt.close()

def loadData(pfad_data, pfad_truth,n,start):
    data = from_binary(pfad_data)
    data_graphs = []
    if len(data) == 1:
        data = data[0]
    truth = from_binary(pfad_truth)
    truth = np.reshape(truth,(len(data),8000))
    if n > len(data):
        n = len(data)
        print("ERROR: The dataset is too short!")
    data = np.array(data[:n])
    dat = np.ones((n,600,51))
    dat[:,:,:50] = data[:n]
    list = []
    for i in range(len(data)):
        list.append((dat[i], truth[i]))
    return list

def loadSplit(pfad_data, pfad_truth,n1,n2):
    data = from_binary(pfad_data)
    data_graphs = []
    if len(data) == 1:
        data = data[0]
    truth = from_binary(pfad_truth)
    truth = np.reshape(truth,(len(data),8000))
    if n1+n2 > len(data):
        print("ERROR: The dataset is too short! Length is %.2f" % len(data))
    data = data[:n1+n2]
    list = []
    for i in range(len(data)):
        list.append((data[i], truth[i]))
    return list[:n1], list[n1:]



class MyData(Dataset):
    def __init__(self,pfad_data, pfad_truth,n,start):
        self.dataset = loadData(pfad_data, pfad_truth,n,start)

    def __getitem__(self, item):
        event, label  = self.dataset[item]
        return event, label

    def __len__(self):
        return len(self.dataset)

class MyData_split(Dataset):
    def __init__(self,pfad_data, pfad_truth,n1,n2):
        self.data_train = loadSplit(pfad_data, pfad_truth,n1,n2)[0]
        self.data_vali = loadSplit(pfad_data, pfad_truth,n1,n2)[1]

    def return_train(self):
        return self.data_train

    def return_vali(self):
        return self.data_vali

# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels


def giveDataloader(p_train_data,p_train_truth,batch_size_t,n,start):
    return DataLoader(MyData(p_train_data,p_train_truth,n,start),batch_size = batch_size_t,  shuffle=True)#,collate_fn=collate)

def giveDataloaders(p_data,p_truth,batch_size_t,batch_size_v,n_t,n_v):
    shuffle = True
    data = MyData_split(p_data,p_truth,n_t,n_v)
    train_loader = DataLoader(data.return_train(), batch_size = batch_size_t,  shuffle=shuffle)
    vali_loader = DataLoader(data.return_vali(), batch_size = batch_size_v,  shuffle=shuffle)
    return train_loader, vali_loader



