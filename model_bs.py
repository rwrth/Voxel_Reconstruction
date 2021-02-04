import dgl
from dgl import backend as B
from scipy import sparse
#from dgl.transform import knn_graph
import torch.nn as nn
import torch
from dgl.nn.pytorch import EdgeConv, GraphConv
from dgl.transform import knn_graph
from dgl.nn.pytorch.glob import MaxPooling
import torch.nn.functional as F
import numpy as np
import gc
import dgl.function as fn
from torch.nn.modules.module import Module
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import ot
from geomloss import SamplesLoss
from torch.autograd import Variable



class Net(nn.Module):
    def __init__(self, n_feats_fc, in_feats_g, Dropout):
        super(Net, self).__init__()
        self.b1 = EdgeConv(54, 54)
        self.b2 = EdgeConv(54, 54)
        self.b3 = EdgeConv(54, 54)
        self.b4 = EdgeConv(54, 54)
        self.b5 = EdgeConv(54, 54)
        self.b6 = EdgeConv(54, 54)
        self.b7 = EdgeConv(54, 54)
        self.b8 =  EdgeConv(54, 54)
        
        self.a1 = EdgeConv(54, 54)
        self.a2 = EdgeConv(54, 54)
        self.a3 = EdgeConv(54, 54)
        self.a4 = EdgeConv(54, 54)
        self.a5 = EdgeConv(54, 25)
        self.a6 = EdgeConv(25, 25)
        self.a7 = EdgeConv(25, 25)
        self.a8 = EdgeConv(25, 25)
        self.a9 = EdgeConv(25, 25)
        self.a10 = EdgeConv(25, 25)
        self.a11 = EdgeConv(25, 25)
        self.a12 = EdgeConv(25, 25)
        self.a13 = EdgeConv(25, 25)
        self.a14 = EdgeConv(25, 25)
        self.a15 = EdgeConv(25, 1)
        
        self.drop = nn.Dropout(0.5)



    def forward(self, graph, graph_vox, inputs, bs, koor):
        zeros = torch.tensor(np.zeros((1,8000,51)))
        new_inputs = torch.cat((torch.cat((inputs, zeros), dim = 1),  torch.reshape(koor,(1,8600,3))), dim = 2)
        new_inputs = torch.reshape(new_inputs.float(),(8600,54))
        feat = self.b1(graph, new_inputs)
        feat = self.b2(graph, feat)
        feat = self.b3(graph, feat)
        feat = self.b4(graph, feat)
        feat = self.b5(graph, feat)
        feat = self.b6(graph, feat)
        feat = self.b7(graph, feat)
        feat = self.b8(graph, feat)
        
        feat = self.drop(feat[600:,:])
        feat = self.a1(graph_vox, feat)
        feat = self.a2(graph_vox, feat)
        feat = self.a3(graph_vox, feat)
        
        feat = self.drop(feat)  
        feat = self.a4(graph_vox, feat)
        feat = self.a5(graph_vox, feat)     
        feat = self.a6(graph_vox, feat)
        
        feat = self.drop(feat)  
        feat = self.a7(graph_vox, feat)      
        feat = self.a8(graph_vox, feat)
        feat = self.a9(graph_vox, feat)
 
        feat = self.drop(feat)  
        feat = self.a10(graph_vox, feat)
        feat = self.a11(graph_vox, feat)       
        feat = self.a12(graph_vox, feat)
        feat = self.a13(graph_vox, feat)
        feat = self.a14(graph_vox, feat)  
        feat = self.a15(graph_vox, feat)       

        out = feat.flatten()

        del inputs, feat ,new_inputs, zeros
        gc.collect()
        return torch.relu(out.view(8000).double())
        
    def giveGraphs(self, batch_size, voxel_pos):
        p2v = np.load("data/p2v_spec.npy",  allow_pickle=True).tolist()
        p2v = [item for sublist in p2v for item in sublist]
        p2p = np.load("data/p2p.npy",  allow_pickle=True).tolist()
        p2p = [item for sublist in p2p for item in sublist]
        v2v = np.load("data/v2v.npy", allow_pickle=True).tolist()
        v2v = [item for sublist in v2v for item in sublist]
        v2v_6 = np.load("data/v2v_6.npy", allow_pickle=True).tolist()
        v2v_6 = [item for sublist in v2v_6 for item in sublist]
        G_vox = dgl.graph(v2v)
        G_vox = dgl.add_self_loop(G_vox)
        
        graph_data = {('PMT', 'p2v', 'vox'): p2v, ('vox', 'v2v', 'vox'): v2v}
        g = dgl.heterograph(graph_data)
        g = dgl.to_homogeneous(g)
        g = dgl.add_self_loop(g)
        G  = dgl.batch([ g for i in range(batch_size)])
        return G, G_vox


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

def loss_nachbarn(p):
    p = np.reshape(p.detach().numpy(),(20,20,20))
    full = np.where(p > 1)
    sort = np.sort(full)
    d = (np.abs(np.sum(np.abs(sort[:,:-1]-sort[:,1:]))-len(sort[0])+1)/np.max([len(sort[0]),1]))
    return d

def loss_line(t, p, koor):
    koor = koor.numpy()
    tT = (t.detach().numpy() > 1)
    full = np.where(tT[0] == True)[0].tolist()
    obs_points = koor[full]
    start = [np.mean(obs_points[:,0]),np.mean(obs_points[:,1]),np.mean(obs_points[:,2])]
    C_x = np.cov(obs_points.T)
    eig_vals, eig_vecs = np.linalg.eigh(C_x)
    max_eig_val_index = np.argmax(eig_vals)
    direction = eig_vecs[:, max_eig_val_index]
    pT = (p.detach().numpy() > 1)
    fullp = np.where(pT[0] == True)[0].tolist()
    sum = 10000
    if len(fullp) > 1:
        points = koor[fullp]
        starts = np.full((points.shape[0],3),start)
        fullung = p.detach().numpy().flatten()[fullp]
        dir = np.full((points.shape[0],3),direction)
        sum = np.mean(fullp * np.linalg.norm(np.cross(dir, starts-points), axis =1)/np.linalg.norm(dir, axis =1))
    return sum 

#erst Klasse Leer
def myCrossEntropy(pred,t):
    CE_loss = nn.CrossEntropyLoss(torch.tensor([0.001,1]))
    p = torch.clamp(pred,0,1)
    p_classes = torch.ones((8000,2))
    p_classes[:,0] = 1-p
    p_classes[:,1] = p
    CE_l = CE_loss(p_classes, torch.clamp(t,0,1).long())
    return CE_l
    
class Where_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred):
        #koor = torch.reshape(koor, (20,20,20,3))
        y = torch.nonzero(pred, as_tuple= False).float()
        ctx.save_for_backward(pred)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        print("HERE In Backward")
        # Here we must handle None grad_output tensor. In this case we
        # can skip unnecessary computations and just return None.
        pred, = ctx.saved_tensors
        if grad_output is None:
            print("grad_output is Nulli")
            return None, None
            
        output = torch.zeros([20,20,20])
        pos = torch.nonzero(pred,as_tuple= False)
        
        ###
        pos_x_0 = pos[torch.where(grad_output[:,0]>0)]
        pos_x_0[:,0] += 1
        pos_x_0 = torch.clamp(pos_x_0, max = 19)
        output[[pos_x_0[:,0],pos_x_0[:,1],pos_x_0[:,2]]] += grad_output[torch.where(grad_output[:,0]>0)][:,0]
        
        pos_x_1 = pos[torch.where(grad_output[:,0]<0)]
        pos_x_1[:,0] -= 1
        pos_x_1 = torch.clamp(pos_x_1, min = 0, max = 19)
        output[[pos_x_1[:,0],pos_x_1[:,1],pos_x_1[:,2]]] += np.abs(grad_output[torch.where(grad_output[:,0]<0)][:,0])
        ###
        pos_y_0 = pos[torch.where(grad_output[:,1]>0)]
        pos_y_0[:,1] += 1
        pos_y_0 = torch.clamp(pos_y_0, min = 0, max = 19)
        output[[pos_y_0[:,0],pos_y_0[:,1],pos_y_0[:,2]]] += grad_output[torch.where(grad_output[:,1]>0)][:,1]
        
        pos_y_1 = pos[torch.where(grad_output[:,1]<0)]
        pos_y_1[:,1] -= 1
        pos_y_1 = torch.clamp(pos_y_1, min = 0, max = 19)
        output[[pos_y_1[:,0],pos_y_1[:,1],pos_y_1[:,2]]] += np.abs(grad_output[torch.where(grad_output[:,1]<0)][:,1])       
        ###
        pos_z_0 = pos[torch.where(grad_output[:,2]>0)]
        pos_z_0[:,2] += 1
        pos_z_0 = torch.clamp(pos_z_0, min = 0, max = 19)
        output[[pos_z_0[:,0],pos_z_0[:,1],pos_z_0[:,2]]] += grad_output[torch.where(grad_output[:,2]>0)][:,2]
        
        pos_z_1 = pos[torch.where(grad_output[:,2]<0)]
        pos_z_1[:,2] -= 1
        pos_z_1 = torch.clamp(pos_z_1, min = 0, max = 19)
        output[[pos_z_1[:,0],pos_z_1[:,1],pos_z_1[:,2]]] += np.abs(grad_output[torch.where(grad_output[:,2]<0)][:,2])        

        del pred, grad_output, pos, pos_x_0, pos_x_1, pos_y_0, pos_y_1,  pos_z_0, pos_z_1
        gc.collect()
        return 0.75*output, None
        
        
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
class MyLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, sigma,a):
        super(MyLoss, self).__init__()
        self.sigma = sigma
        self.a = a
        self.loss = SamplesLoss("sinkhorn", p=1, blur=.1, scaling=0.8, verbose=True)#, backend="tensorized")
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()       

    def forward(self, p, t, koor):
        cut = 100
        pred = p.view(20,20,20)
        tru = t.view(20,20,20)
        a = Where_grad()
        x = torch.nonzero(tru,as_tuple = False).float()
        weight1 = tru[torch.nonzero(tru, as_tuple = True)].float()#/torch.sum(tru)
        y = a.apply(torch.clamp(pred-cut,0))
        pred_full = torch.nonzero(torch.clamp(pred-cut, min = 0),as_tuple = True)
        weight2 = pred[pred_full].float()#/torch.sum(tru)
        if y.size() == torch.Size([0, 3]):
            print("Cut = 1 ")
            y = a.apply(torch.clamp(pred-1, min = 0))
            pred_full = torch.nonzero(torch.clamp(pred-1, min = 0),as_tuple = True)
            weight2 = pred[pred_full].float()#/torch.sum(tru)
            if y.size() == torch.Size([0, 3]):
                a = 1
                y = torch.reshape(torch.ones(3*a)*100,(3,a)).float().t()
                weight2 = torch.zeros(a).float()#*(-torch.sum(tru))
                y.requires_grad = True
        Wass_xy = torch.abs(self.loss(weight1, x, weight2, y))*0.625
        CELoss = myCrossEntropy(p,t[0])*10**8*1.1
        b = torch.nonzero(torch.clamp(t[0],0))
        c = list(range(8000))
        for b1 in b:
            c.remove(b1)
        loss_spur = self.mse(p[b],t[0][b])*10000
        loss_b = self.mse(p[c],t[0][c])*25000
        d = torch.nonzero(torch.clamp(t[0]-2000,0))
        loss_max = self.mse(p[d],t[0][d])*1000
        #print(CELoss, Wass_xy, loss_b,loss_spur,loss_max)
        #loss = self.mse(p,t[0])
        return Wass_xy + CELoss + loss_b +loss_spur+loss_max
