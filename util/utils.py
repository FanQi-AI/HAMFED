import torch
import numpy as np
import copy
import torch.nn as nn


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten2(model):
    no_model_agg = ['no_model_mlp.weight','no_model_mlp.bias','BN2.weight','BN2.bias','no_model_mlp2.weight','no_model_mlp2.bias','BN4.weight','BN4.bias']
    params = []
    for k in model:
        if k in no_model_agg:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten1(model):
    model_agg    = ['model_mlp.weight','model_mlp.bias','BN1.weight','BN1.bias','model_mlp2.weight','model_mlp2.bias','BN3.weight','BN3.bias']
    params = []
    for k in model:
        if k in model_agg:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round))) 
    index_clientid = list(nets_this_round.keys())    #[20, 14, 15, 11, 28, 2, 23, 1, 24, 17, 19, 3, 16, 25, ...]
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]     
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":  
                diff =  torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))   #[1.429833]之间求相似度
                if diff >  0.9:
                    diff =  1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif  similarity_matric == "no_model":
                diff =  torch.nn.functional.cosine_similarity(weight_flatten2(dw[index_clientid[i]]).unsqueeze(0), weight_flatten2(dw[index_clientid[j]]).unsqueeze(0))
                if diff > 0.9:
                    diff =  1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    x1,x2=[],[] 
    for i in range(len(nets_this_round)):
        for _ in range(8):
            x1.append(weight_flatten1(dw[index_clientid[i]]).tolist())
            x2.append(weight_flatten2(dw[index_clientid[i]]).tolist())
            

    
    return model_similarity_matrix,x1,x2


from util.config import *
def agg_mutimoel_data(selected_clients_id,clients_models,device,dataloaders):
    metrics=[]

    for c in selected_clients_id:
        model=clients_models['clients'][c]
        model.eval()
        model.to(device)
        
        total = 0
        correct = 0
        a_acc_class = [0  for _ in range(NUM_CLASSES)]
        with torch.no_grad():
            for data in dataloaders['common'][0]:    
                x=data[0].to(device).to(torch.float32)
                y=data[1].to(device)
            
                output,model_fea,nomodel_fea=model(x,x)
                pred_class = torch.max(output.data, 1)[1]
                total += y.size(0)
                correct += (pred_class == y).sum().item()
                for i in range(y.size(0)):
                    if y[i] == pred_class[i]:
                        a_acc_class[y[i]] += 1
        a_acc=100 * correct / total

        
        total = 0
        correct = 0
        v_acc_class = [0  for _ in range(NUM_CLASSES)]
        with torch.no_grad():
            for data in dataloaders['common'][1]:    
                x=data[0].to(device).to(torch.float32)
                y=data[1].to(device)
            
                output,model_fea,nomodel_fea=model(x,x)
                pred_class = torch.max(output.data, 1)[1]
                total += y.size(0)
                correct += (pred_class == y).sum().item()
                for i in range(y.size(0)):
                    if y[i] == pred_class[i]:
                        v_acc_class[y[i]] += 1
        v_acc=100 * correct / total

        #t
        total = 0
        correct = 0
        t_acc_class = [0  for _ in range(NUM_CLASSES)]
        with torch.no_grad():
            for data in dataloaders['common'][2]:    
                x=data[0].to(device).to(torch.float32)
                y=data[1].to(device)
            
                output,model_fea,nomodel_fea=model(x,x)
                pred_class = torch.max(output.data, 1)[1]
                total += y.size(0)
                correct += (pred_class == y).sum().item()
                for i in range(y.size(0)):
                    if y[i] == pred_class[i]:
                        t_acc_class[y[i]] += 1
        t_acc=100 * correct / total

        if a_acc>=v_acc  and a_acc>=t_acc :
            metrics.append(np.array(a_acc_class)/100)
        elif v_acc>=a_acc and v_acc>=t_acc :
            metrics.append(np.array(v_acc_class)/100)
        else:
            metrics.append(np.array(t_acc_class)/100)
        model.cpu()
    return metrics

def agg_mutimoel_data2m(selected_clients_id,clients_models,device,dataloaders):
    metrics=[]
    mutimodal_prototype_clients=[ [] for _ in range(Mutimodel_num) ]
    for c in selected_clients_id:
        modal_a_id_pro=[[]for _ in range(NUM_CLASSES)]
        model=clients_models['clients'][c]
        model.eval()
        model.to(device)
        
        total = 0
        correct = 0
        a_acc_class = [0  for _ in range(NUM_CLASSES)]
        with torch.no_grad():
            for data in dataloaders['common'][0]:    #a
                x=data[0].to(device).to(torch.float32)
                y=data[1].to(device)
            
                output,model_fea,nomodel_fea=model(x,x)
                pred_class = torch.max(output.data, 1)[1]
                total += y.size(0)
                correct += (pred_class == y).sum().item()
                for i in range(y.size(0)):
                    if y[i] == pred_class[i]:
                        a_acc_class[y[i]] += 1
                    label=y[i].item()
                    modal_a_id_pro[label].append(nomodel_fea[i].cpu().numpy().tolist())
        a_acc=100 * correct / total

        modal_v_id_pro=[[]for _ in range(NUM_CLASSES)]
        
        total = 0
        correct = 0
        v_acc_class = [0  for _ in range(NUM_CLASSES)]
        with torch.no_grad():
            for data in dataloaders['common'][1]:    
                x=data[0].to(device).to(torch.float32)
                y=data[1].to(device)
            
                output,model_fea,nomodel_fea=model(x,x)
                pred_class = torch.max(output.data, 1)[1]
                total += y.size(0)
                correct += (pred_class == y).sum().item()
                for i in range(y.size(0)):
                    if y[i] == pred_class[i]:
                        v_acc_class[y[i]] += 1
                    label=y[i].item()
                    modal_v_id_pro[label].append(nomodel_fea[i].cpu().numpy().tolist())
        v_acc=100 * correct / total
        mutimodal_prototype_clients[0].append(modal_a_id_pro)
        mutimodal_prototype_clients[1].append(modal_v_id_pro)
        if a_acc>=v_acc  :
            metrics.append(np.array(a_acc_class)/100)
        else:
            metrics.append(np.array(v_acc_class)/100)
        model.cpu()
    mutimodal_prototype_clients=np.mean(np.array(mutimodal_prototype_clients), axis=-2)
    return metrics , mutimodal_prototype_clients 

def v_e(clusters1,clusters2):#[[1,2,3],[0]]
    v1,e1=[],[]
    num=0
    for z in clusters1:
        for j in z:
            v1.append(j)
            e1.append(num)
        num +=1

    v2,e2=[],[]
    num=0
    for z in clusters2:
        for j in z:
            v2.append(j)
            e2.append(num)
        num +=1

    return v1,e1,v2,e2


def edgnn_loss(w1,w2,clusters,clusters2,selected_clients_id,clients_models,device,dataloaders,writer,com,ep):
    model_agg    = ['model_mlp.weight','model_mlp.bias','BN1.weight','BN1.bias','model_mlp2.weight','model_mlp2.bias','BN3.weight','BN3.bias']

    ZU1=[]
    ZU2=[]
    for ids in clusters:
        softmax=nn.Softmax( dim=0)
        soft_weight=softmax(w1[ids]).cpu()
        model_state = copy.deepcopy(clients_models['clients'][selected_clients_id[ids[0]]].state_dict())
        for key in model_agg:
            model_state[key] = model_state[key] * soft_weight[0][0]
        for i in range(1,len(ids)):
            c=selected_clients_id[ids[i]]
            for key in model_agg:
                model_state[key] = model_state[key] + clients_models['clients'][c].state_dict()[key] *soft_weight[i][0]
        ZU1.append(model_state)

    for ids in clusters2:
        # soft_weight=nn.Softmax( w2[ids])
        softmax=nn.Softmax( dim=0)
        soft_weight=softmax(w2[ids]).cpu()
        model_state = copy.deepcopy(clients_models['clients'][selected_clients_id[ids[0]]].state_dict())
        for key in model_state:
            if key not in model_agg:
                model_state[key] = model_state[key] * soft_weight[0][0]
        for i in range(1,len(ids)):
            c=selected_clients_id[ids[i]]
            for key in model_state:
                if key not in model_agg:
                    model_state[key] = model_state[key] + clients_models['clients'][c].state_dict()[key] *soft_weight[i][0]
        ZU2.append(model_state)
    
    total_acc=[]
    del model_state
    
    model=copy.deepcopy(clients_models['clients'][0])
    for id in range(len(selected_clients_id)):
        c=selected_clients_id[id]
        
        yv1=[]
        yv2=[]
        for a in range(len(clusters)):
            if id in clusters[a]:
                yv1.append(a)
        for a in range(len(clusters2)):
            if id in clusters2[a]:
                yv2.append(a)
        

       
        weght1=0.5
        weght2=1/len(yv1)/2
        weght3=1/len(yv2)/2
        model_state = copy.deepcopy(clients_models['clients'][c].state_dict())
        for key in model_state:
            model_state[key] = model_state[key] * weght1
        for i in range(len(yv1)):
            for key in model_agg:
                model_state[key] = model_state[key] + ZU1[yv1[i]][key] *weght2
        for i in range(len(yv2)):
            for key in model_state:
                if key not in model_agg:
                    model_state[key] = model_state[key] + ZU2[yv2[i]][key] *weght3
        model.load_state_dict(model_state,strict=False)
        
        
        model.eval().to(device)
        total = 0
        correct = 0 
        with torch.no_grad():
            for data in dataloaders['clients_test'][c]:
                x=data[0].to(device).to(torch.float32)
                y=data[1].to(device)
            
                output,_,_=model(x,x)
                pred_class = torch.max(output.data, 1)[1]
                total += y.size(0)
                correct += (pred_class == y).sum().item()
        total_acc.append(correct / total)
    del model
    meanacc=sum(total_acc)/len(total_acc)
   
    
    myloss=MyLoss()

    return myloss(meanacc)

class MyLoss(nn.Module):
    def forward(self,acc):
        loss=1-64**(acc-1)
        a=torch.FloatTensor([loss])
        a.requires_grad=True
        return  a
    
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
        
def update_clients_models(w1,w2,clusters,clusters2,selected_clients_id,clients_models):
    model_agg    = ['model_mlp.weight','model_mlp.bias','BN1.weight','BN1.bias','model_mlp2.weight','model_mlp2.bias','BN3.weight','BN3.bias']

    ZU1=[]
    ZU2=[]
    for ids in clusters:
        softmax=nn.Softmax( dim=0)
        soft_weight=softmax(w1[ids]).cpu()
        model_state = copy.deepcopy(clients_models['clients'][selected_clients_id[ids[0]]].state_dict())
        for key in model_agg:
            model_state[key] = model_state[key] * soft_weight[0][0]
        for i in range(1,len(ids)):
            c=selected_clients_id[ids[i]]
            for key in model_agg:
                model_state[key] = model_state[key] + clients_models['clients'][c].state_dict()[key] *soft_weight[i][0]
        ZU1.append(model_state)

    for ids in clusters2:
        # soft_weight=nn.Softmax( w2[ids])
        softmax=nn.Softmax( dim=0)
        soft_weight=softmax(w2[ids]).cpu()
        model_state = copy.deepcopy(clients_models['clients'][selected_clients_id[ids[0]]].state_dict())
        for key in model_state:
            if key not in model_agg:
                model_state[key] = model_state[key] * soft_weight[0][0]
        for i in range(1,len(ids)):
            c=selected_clients_id[ids[i]]
            for key in model_state:
                if key not in model_agg:
                    model_state[key] = model_state[key] + clients_models['clients'][c].state_dict()[key] *soft_weight[i][0]
        ZU2.append(model_state)
    
    
    del model_state
    
   
    for id in range(len(selected_clients_id)):
        c=selected_clients_id[id]
        
        yv1=[]
        yv2=[]
        for a in range(len(clusters)):
            if id in clusters[a]:
                yv1.append(a)
        for a in range(len(clusters2)):
            if id in clusters2[a]:
                yv2.append(a)
        
        weght1=0.5
        weght2=1/len(yv1)/2
        weght3=1/len(yv2)/2
        model_state = copy.deepcopy(clients_models['clients'][c].state_dict())
        for key in model_state:
            model_state[key] = model_state[key] * weght1
        for i in range(len(yv1)):
            for key in model_agg:
                model_state[key] = model_state[key] + ZU1[yv1[i]][key] *weght2
        for i in range(len(yv2)):
            for key in model_state:
                if key not in model_agg:
                    model_state[key] = model_state[key] + ZU2[yv2[i]][key] *weght3

        

        clients_models['clients'][c].load_state_dict(model_state,strict=False)
    