
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np



class MELD_MODEL(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MELD_MODEL, self).__init__()
        self.layer_input = nn.Linear(dim_in, 2048) 
        self.BN1 = nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True)
        self.BN2 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
        self.BN3 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU() 
        #self.dropout = nn.Dropout(p=0.2) 
        self.fc1 = nn.Linear(2048, 4096)  
        self.fc2 = nn.Linear(4096, 1024)  
        self.layer_out = nn.Linear(1024, dim_out)  
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = ['layer_input.weight', 'layer_input.bias',
                            'fc1.weight', 'fc1.bias',
                            'fc2.weight', 'fc2.bias',
                            'layer_out.weight', 'layer_out.bias'
                            ]

    def forward(self, x1,x2):
        x = x.view(-1,2048)
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return self.softmax(x)
    
class Attention(nn.Module):
    def __init__(self, model_dim, no_model_dim, classes_num=7, layers='256,128', dropout=0.3):
        super(Attention, self).__init__()
        layers_list = list(map(lambda x: int(x), layers.split(',')))

        self.model_mlp=nn.Linear(model_dim, layers_list[0])
        self.BN1 = nn.BatchNorm1d(layers_list[0], eps=1e-05, momentum=0.1, affine=True)

        self.no_model_mlp=nn.Linear(no_model_dim, layers_list[0])
        self.BN2 = nn.BatchNorm1d(layers_list[0], eps=1e-05, momentum=0.1, affine=True)
        
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(dropout)

        self.model_mlp2=nn.Linear(layers_list[0], layers_list[1])
        self.BN3 = nn.BatchNorm1d(layers_list[1], eps=1e-05, momentum=0.1, affine=True)
        self.no_model_mlp2=nn.Linear(layers_list[0], layers_list[1])
        self.BN4 = nn.BatchNorm1d(layers_list[1], eps=1e-05, momentum=0.1, affine=True)
        hiddendim = layers_list[-1] * 2
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 2)
        self.fc_out_1 = nn.Linear(layers_list[-1], classes_num)
        
        self.softmax = nn.Softmax(dim=1)
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, audio_feat, no_audio_feat):
        audio_hidden=self.model_mlp(audio_feat)
        audio_hidden=self.BN1(audio_hidden)
        audio_hidden=self.relu(audio_hidden)
        audio_hidden=self.drop(audio_hidden)

        audio_hidden=self.model_mlp2(audio_hidden)
        audio_hidden=self.BN3(audio_hidden)
        audio_hidden=self.relu(audio_hidden)
        audio_hidden=self.drop(audio_hidden)  
        model_fea=audio_hidden


        no_audio_hidden=self.no_model_mlp(no_audio_feat)
        no_audio_hidden=self.BN2(no_audio_hidden)
        no_audio_hidden=self.relu(no_audio_hidden)
        no_audio_hidden=self.drop(no_audio_hidden)

        no_audio_hidden=self.no_model_mlp2(no_audio_hidden)
        no_audio_hidden=self.BN4(no_audio_hidden)
        no_audio_hidden=self.relu(no_audio_hidden)
        no_audio_hidden=self.drop(no_audio_hidden) 
        no_model_fea=no_audio_hidden


        multi_hidden1 = torch.cat([audio_hidden, no_audio_hidden], dim=1) 
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) 

        multi_hidden2 = torch.stack([audio_hidden, no_audio_hidden], dim=2) 
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        
        return  self.softmax(emos_out),model_fea,no_model_fea
    

class EquivSetGNN(nn.Module):
    def __init__(self, num_features_dim,drop=0.1,hid_dim=128,nlayer=2,activation='relu'):
        super().__init__()

        self.nlayer=nlayer
        self.dropout = nn.Dropout(drop) # 0.2 is chosen for GCNII
        self.lin_in = torch.nn.Linear(num_features_dim, hid_dim)
        self.conv = EquivSetConv(hid_dim,hid_dim, mlp1_layers=1, mlp2_layers=1,mlp3_layers=1, alpha=0.5, aggr='mean',dropout=drop, normalization='ln', input_norm=True)

        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu':nn.PReLU()}
        self.act = act[activation]
        self.classifier = MLP(in_channels=hid_dim,hidden_channels=64,out_channels=1,num_layers=2,dropout=drop,Normalization='ln',InputNorm=False)

    def forward(self,x1,v1,e1,x2,v2,e2):
        x1 = self.dropout(x1)
        x1= F.relu(self.lin_in(x1))
        x1_0=x1
        for i in range(self.nlayer):
            x1 = self.dropout(x1)
            x1 = self.conv(x1, v1, e1, x1_0)
            x1 = self.act(x1)
        x1 = self.dropout(x1)
        x1 = self.classifier(x1)

        x2 = self.dropout(x2)
        x2= F.relu(self.lin_in(x2))
        x2_0=x2
        for i in range(self.nlayer):
            x2 = self.dropout(x2)
            x2 = self.conv(x2, v2, e2, x2_0)
            x2 = self.act(x2)
        x2 = self.dropout(x2)
        x2 = self.classifier(x2)

        return x1,x2




class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
        mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features+out_features, out_features, out_features, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout
    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :] # [nnz, C]
        Xe = torch_scatter.scatter(Xve, torch.tensor(edges).to(X.device), dim=-2, reduce=self.aggr) # [E, C], reduce is 'mean' here as default
        
        Xev = Xe[..., edges, :] # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, torch.tensor(vertex).to(X.device), dim=-2, reduce=self.aggr, dim_size=N) # [N, C]

        X = Xv

        X = (1-self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels # first normalization
        flops += num_samples * self.in_channels * self.hidden_channels # first linear layer
        flops += num_samples * self.hidden_channels # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += num_samples * self.out_channels * self.hidden_channels # last linear layer

        return flops
#test
if __name__ == '__main__':
    
    
    
    dw = []   
    
    dw.append({key : torch.zeros_like(value) for key, value in model.named_parameters()})


    

    
    model_agg    = ['model_mlp.weight','model_mlp.bias','BN1.weight','BN1.bias','model_mlp2.weight','model_mlp2.bias','BN3.weight','BN3.bias']
    no_model_agg = ['no_model_mlp.weight','no_model_mlp.bias','BN2.weight','BN2.bias','no_model_mlp2.weight','no_model_mlp2.bias','BN4.weight','BN4.bias']
   
    print(model.model_mlp.weight)
    print(model2.model_mlp.weight)
    print(model3.model_mlp.weight)

    import copy
    model_state = copy.deepcopy(model.state_dict())
    for key in model_state:
        if key in model_agg:
            model_state[key] = model_state[key] /2
    for key in model_state:
        if key in model_agg:
            model_state[key] = model_state[key] + model2.state_dict()[key]/2
    model3_state=model3.state_dict()
    # for key in model_state:
    for key in model_agg:
        model3_state[key] = model_state[key]
    model3.load_state_dict(model3_state,strict=False)
    print(model.model_mlp.weight)
    print(model2.model_mlp.weight)
    print(model3.model_mlp.weight)
    pass



    
    

    
