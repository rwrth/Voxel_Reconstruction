from model_bs import Net, MyLoss
from evaluate_bs import validate_loss, performance
from prepareData_bs import giveDataloader, giveDataloaders
from make_log_bs import log, save_model, str2bool

import torch.nn as nn
import torch
import dgl
import numpy as np
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import *
import argparse
import gc
import torch.autograd.gradcheck


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('save', help='Should this run be saved?')
parser.add_argument('comment', help='Commet about the architecture or something.')
args = parser.parse_args()
save = str2bool(args.save)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

p_train_data = "newData/Set13_data.bin"
p_train_truth =  "newData/Set13_voxel_truth.bin"

p_vali_data = "newData/Set2_data.bin"
p_vali_truth = "newData/Set2_voxel_truth.bin"

n_train = 100
n_vali = 100
batch_size_t = batch_size_v = 1
epochs = 10

learning_rate = 0.00005
sigma = 0.1
Dropout = 0.15
a = [1,0.25,1]

torch.manual_seed(0)
np.random.seed(0)

#train_loader, vali_loader = giveDataloaders(p_data,p_truth,batch_size_t,batch_size_v,n_train,n_vali)
positions_pmt = np.load("data/positions.npy")
positions_vox = torch.tensor(np.reshape(np.load("data/Voxelkoordinaten.npy"),(8000,3)))
voxel_pos = torch.tensor(np.load("data/Voxelkoordinaten.npy")).reshape((8000,3))
koor = torch.tensor(np.vstack((positions_pmt, positions_vox)))
print(positions_vox.shape, positions_pmt.shape, koor.shape)
start = torch.tensor(positions_pmt).clone().detach()
train_loader =   giveDataloader(p_train_data, p_train_truth, batch_size_t, n_train, start)
vali_loader =  giveDataloader(p_vali_data, p_vali_truth, batch_size_v, n_vali, start)
print("%i train events" %n_train)
print("%i validation events" %n_vali)


net = Net(50,10, Dropout)
net = net.float().to(device)
graph, graph_vox = net.giveGraphs(batch_size_t, voxel_pos)
#net.train()
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
loss_f =  MyLoss(sigma, a)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#check gradients of the Loss function 
from torch.autograd import gradcheck
if False: 
    input, label = next(iter(train_loader))
    prediction = net(graph, graph_vox, input, batch_size_t, koor).double()
    res = gradcheck(loss_f, (prediction, label.double(), koor[600:]),  eps=1e-2,check_undefined_grad = True, raise_exception = True)
    print("Gradcheck of Lossfunction gives:  " +str(res))

if save:
    my_log = log()
    log.writelog_file_start(my_log, args.comment, p_train_data, p_vali_data, batch_size_t, batch_size_v, learning_rate, trainable_params, sigma, Dropout)
    writer = SummaryWriter(my_log.give_dir())
else:
    writer = 0

all_learned = 0
overfit = 0
dif_all =[]
dif_std_all = []
all_m_Vloss = []
all_Vloss = []
all_m_loss = []
all_loss = []
richtig = []
for epoch in range(epochs):
    print('Epoch %d' %epoch)
    m_loss = []
    m_Vloss = []
    for i, (input, label) in enumerate( tqdm(train_loader)):
        optimizer.zero_grad()
        prediction = net(graph, graph_vox, input, batch_size_t, koor)
        loss = loss_f(prediction, label, koor[600:])
        loss.backward()
        optimizer.step()
        #for p in net.parameters():
        #    print(p.grad.norm()) 
        m_loss.append(loss.item())
        if save:
            writer.add_scalar("loss", loss.item())
        vall_loss = validate_loss(vali_loader, net, graph, graph_vox, batch_size_v, save, writer, loss_f, koor)
        m_Vloss.append(vall_loss.item())
        del input, label, prediction, i, loss, vall_loss
        gc.collect()
    all_m_loss.append(np.mean(m_loss))
    all_loss.append(m_loss)
    all_m_Vloss.append(np.mean(m_Vloss))
    all_Vloss.append(m_Vloss)
    dif, dif_std, r_voll_m, r_voll_s, r_leer_m, r_leer_s = performance(graph,graph_vox, vali_loader, net, koor, batch_size_t)
    richtig.append([r_voll_m, r_voll_s, r_leer_m, r_leer_s])
    dif_all.append(dif)
    dif_std_all.append(dif_std)
    print("Anteil der richtig gefuellten Voxel " +str( r_voll_m) +"+/-" +str( r_voll_s))
    print("Anteil der richtig leer Voxel " +str(r_leer_m) +"+/-" +str(r_leer_s))
    print('Loss: %.4f  | Validation Loss: %.4f' % (all_m_loss[-1], all_m_Vloss[-1]))
    print("mittlere Differenz der Summen: %.2f +/- %.2f " %(dif, dif_std))
    del dif, dif_std, m_loss, m_Vloss, r_voll_s, r_leer_m, r_leer_s
    gc.collect()
    if len(all_m_loss) > 1:
        if round(all_m_loss[-1],1)==round(all_m_loss[-2],1):
            all_learned +=1
            if all_learned > 2:
                print("Break up training because mean loss does not change, twice.")
                epochs = epoch+1
                break
        else:
            all_learned = 0
        if all_m_Vloss[-1]*0.98 > all_m_loss[-1]:
            overfit +=1
            if overfit >=2:
                print("Break up training because of overfitting.")
                epochs = epoch+1
                break
        else:
            overfit = 0
    if save:
    	save_model(net, my_log.give_dir() + "/models", epoch)


if save:
    writer.close()
    log.writelog_file_end(my_log, epochs, all_loss, all_Vloss)
    log.save_plots(my_log, epochs, n_train/batch_size_t, np.ravel(all_loss), all_m_loss, np.ravel(all_Vloss), all_m_Vloss, dif_all,dif_std_all,richtig )
