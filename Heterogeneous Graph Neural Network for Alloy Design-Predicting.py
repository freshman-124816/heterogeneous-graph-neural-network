import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import itertools
from torch_geometric.nn import Linear, GATConv, Sequential, to_hetero
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.nn import Linear

import torch
import torch.nn as nn
from torch.nn import Linear

import torch
import torch.nn as nn
from torch.nn import Linear


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(3407)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

#model construction
class G_aggr(torch.nn.Module):
    def __init__(self,layer3):
        super().__init__()
        self.lin=Linear(layer3,2)

    def forward(self, x,allow_unused=True):
        x=torch.cat((x['elements'],x['process']),dim=0)
        x=self.lin(x)
        x=x.mean(dim=0)
        return x

class GNN(nn.Module):
    def __init__(self, add_self_loops, hidden_layer1, hidden_layer2, hidden_layer3):
        super().__init__()
        
        self.conv1 = GATConv((-1, -1), hidden_layer1, negative_slope=0, add_self_loops=add_self_loops)
        self.conv2 = GATConv((-1, -1), hidden_layer2, negative_slope=0, add_self_loops=add_self_loops)
        self.conv3 = GATConv((-1, -1), hidden_layer3, negative_slope=0, add_self_loops=add_self_loops)

    def forward(self, x, edge_index, allow_unused=True):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x

class G_process(nn.Module):
    def __init__(self, metadata, hidden_layer1, hidden_layer2, hidden_layer3):
        super().__init__()
       
        self.encode = GNN(False, hidden_layer1, hidden_layer2, hidden_layer3)
        
        self.encode = to_hetero(self.encode, metadata, aggr='sum').to(device)
        
        self.decode = G_aggr(hidden_layer3)

    def forward(self, x_dict, edge_index_dict, allow_unused=True):
        
        x = self.encode(x_dict, edge_index_dict)
       
        return self.decode(x)

# metadata definitions
metadata = (['elements', 'process'],
           [('elements', 'e', 'elements'),
            ('elements', 'e_p', 'process'),
            ('process', 'p', 'process'),
            ('process', 'rev_e_p', 'elements')])

# data load

data_file='./data.xlsx'#input data file path
data = pd.read_excel(data_file)
data_physical_property_all = data.iloc[:57,1:17]
data_process_number = data.iloc[0,17:-2]

shanchu_list = [44,12,16,26,32,33,34,9,22,0,25,50,4,13,14,24,15,46,17,47,35,38,39,18,19,31,48,55,37,53,
              51,43,21,36,23,49,8,5,7,41,6,28,10,11,42,30,3,45,27,52,54]

data_physical_property_all = data_physical_property_all.drop(shanchu_list)
data_physical_property_all = np.array(data_physical_property_all)
data_physical_property_all = pd.DataFrame(data_physical_property_all)
data_physical_property_new = np.array(data_physical_property_all)

name_list = ['Cu','Ag','Cr','Ca','Co','Ce','La','Mg','Ni','Si','Sn','Sr','Y','Yb','Zn','Zr']
data_physical_property = pd.DataFrame(data_physical_property_new)
data_H = []
alloy_name_dict = {}

#Data preprocessing
for i in range(57, len(data)):
    data_1 = data.iloc[i,1:]
    data_1x = np.array(data_1.iloc[:-2])
    data_1y = np.array(data_1.iloc[-2:])
    E_label = data_1x[:16]==0
    data_1H = HeteroData()
    E_data = []
    alloy_name_list = []
    
    for jj in range(len(E_label)):
        if E_label[jj] == False:
            alloy_name_list.append(name_list[jj])
        alloy_name_dict[i] = alloy_name_list
    
    
    for k in range(len(E_label)):
        if E_label[k] == False:
            E_array = np.concatenate((np.array(data_1x[k]).reshape(-1), np.array(data_physical_property.iloc[:,k])), axis=0)
            E_data.append(E_array)
            
    data_1H['elements'].x = torch.tensor(E_data, dtype=torch.float).to(device)
    data_y = torch.tensor([data_1y[0], data_1y[1]], dtype=torch.float).to(device)
    data_1H['elements'].y = data_y  

   
    E_no_conbine = [i for i in range(len(E_data))]
    E_conbine = list(itertools.combinations(E_no_conbine, 2))
    emelents_band_edge = torch.tensor(E_conbine).reshape(-1, 2).T
    data_1H['elements','e','elements'].edge_index = emelents_band_edge.to(device)  

    
    P_label = data_1x[16:]==0
    P_data = []
    for j in range(len(P_label)):
        if P_label[j] == False:
            P_data.append([data_process_number.iloc[j], data_1x[j+16]])
    if len(P_data) == 0:
        P_data.append([0.0, 0.0])  
    for i in P_data:
        data_1H['process'].x = torch.tensor(P_data, dtype=torch.float).reshape(-1, 2).to(device)
        
    
    data_1H['elements','e_p','process'].edge_index = torch.tensor([[x for x in range(len(E_data))], [0]*len(E_data)]).to(device)

    
    if len(P_data) > 1:
        P_conbine = [[i for i in range(len(P_data)-1)], [i for i in range(1, len(P_data), 1)]]
        process_band_edge = torch.tensor(P_conbine).to(device)
        data_1H['process','p','process'].edge_index = process_band_edge
    else:
        data_1H['process','p','process'].edge_index = torch.tensor([[0], [0]]).to(device)
        
    data_1H = T.ToUndirected()(data_1H)    
    data_H.append(data_1H)

# model initialization
model_file='./model.pth' #input model file path
model=torch.load(model_file)

with torch.no_grad():
    outout=[]
    model.eval()
    for i in range(len(data_H)):
        data=data_H[i]
        out=model(data.x_dict,data.edge_index_dict)
        out=out.cpu().detach().numpy()
        outout.append(out)
    
outout=np.array(outout)
outout=pd.DataFrame(outout)
outout.to_excel('./output.xlsx') #output file path