# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
# Model
from model.model import Attention,EquivSetGNN
import copy
from util.config import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
# Utils
from util.config import *
from tqdm import tqdm
from util.noiid_setting import dirichlet_split_noniid
from util.utils import cal_model_cosine_difference,agg_mutimoel_data2m,v_e,edgnn_loss,update_clients_models,DiffLoss
from enhancer import *
from util.Infonce import PrototypeInfoNCE
# dataset
from datasets import DPIC_DataSet
from datasets import SubsetSequentialSampler
from train import Separation_Loss,test
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
writer = SummaryWriter('/home/logs')
def train(clients_models,dataloaders,epoch,device,clients_data_num):
    optim_clients = [] 
    sched_clients = [] 
    for c in range(CLIENTS*2):
        optim_clients.append(optim.Adam(clients_models['clients'][c].parameters(), lr=LR, weight_decay=WDECAY))
        sched_clients.append(lr_scheduler.MultiStepLR(optim_clients[c], milestones=MILESTONES)) 

    edgnn_op=optim.Adam(clients_models['edgnn'].parameters(), lr=EDGNN_LR, weight_decay=WDECAY)   
    edgnn_lr=lr_scheduler.MultiStepLR(edgnn_op, milestones=MILESTONES)

    optim_sched_clients={'optim':optim_clients,'sched':sched_clients}
    
    global_prototype=[]
    pca = PCA(n_components=96) 
    modal_enhancer=Multimodal_enhancer(mod1_dim=FEADIM, mod2_dim=FEADIM, num_classes=NUM_CLASSES, prototype_dim=256)
    datasetcommon_mutimodal = Common_DataSet(mode='common',path=None)
    common_dataloader_mutimodal=DataLoader(datasetcommon_mutimodal,batch_size=BATCH)
    
    for com in range(COMMUNICATION): 
        
        selected_clients_id = np.random.choice(CLIENTS*Mutimodel_num, int(CLIENTS*Mutimodel_num * RATIO), replace=False)
        for epoch in range(EPOCH): 
            loss_total=train_epoch_client(selected_clients_id,clients_models,optim_sched_clients,dataloaders,device,comm_dataloader=common_dataloader_mutimodal)
        writer.add_scalar('loss', loss_total/total_data_num, com)
        acc_mean=test(selected_clients_id,clients_models,dataloaders,device,writer,com,total_data_num)
        writer.add_scalar('acc_mean', acc_mean, com)

        metrics,mutimodal_prototype_clients=agg_mutimoel_data2m(selected_clients_id,clients_models,device,dataloaders) #[2,16,101,256]

        global_prototype = Prototype_Enhancer(pro_model=modal_enhancer,device=device,mutimodal_prototype_clients=mutimodal_prototype_clients,dataloaders=common_dataloader_mutimodal)
        
        n_clusters = 2
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',random_state=0).fit(metrics)
        
        resp = gmm.predict_proba(metrics) 
        threshold = 0.2
        clusters = [ [] for _ in range(n_clusters) ] 
        for i in range(resp.shape[0]):
            for j in range(resp.shape[1]):
                if resp[i][j]>threshold:
                    clusters[j].append(i)
        print(clusters)
        del gmm,resp,metrics

        initial_global=Attention(FEADIM,FEADIM,NUM_CLASSES,'512,256',drop_para).to('cpu')
        dw = []
        for c in range(len(clients_models['clients'])):
            dw.append({key : torch.zeros_like(value) for key, value in clients_models['clients'][c].named_parameters()})
        nets_this_round = {k: clients_models['clients'][k] for k in selected_clients_id}
        model_difference_matrix, X1 ,X2 = cal_model_cosine_difference(nets_this_round, initial_global.state_dict(), dw, similarity_matric='no_model')   

        ap= AffinityPropagation( affinity="precomputed") 
        ap.fit(model_difference_matrix)
        cluster_centers = ap.cluster_centers_indices_ 
        print(cluster_centers)
        labels = ap.labels_
        clusters2 = [[] for _ in range(len(cluster_centers))]
        for i in range(len(selected_clients_id)):
            clusters2[labels[i]].append(i)
        print(clusters2)
        V1,E1,V2,E2=v_e(clusters,clusters2)

        X1 = pca.fit_transform(np.array(X1))
        X2 = pca.fit_transform(np.array(X2))
        X11=[]
        X22=[]
        for i in range(0,int(CLIENTS*Mutimodel_num * RATIO*8),8):
            X11.append(X1[i])
            X22.append(X2[i])
        X1=torch.Tensor(X11)
        X2=torch.Tensor(X22)
        model=clients_models['edgnn'].to(device) 
        model.train()   
        X1=X1.to(device)
        X2=X2.to(device)
        for ep in range(EDGNN_EPOCH+1):
            edgnn_op.zero_grad()
            w1,w2=model(X1,V1,E1,X2,V2,E2)
            if ep==EDGNN_EPOCH:
                update_clients_models(w1,w2,clusters,clusters2,selected_clients_id,clients_models)
                pass
            else:
                loss=edgnn_loss(w1,w2,clusters,clusters2,selected_clients_id,clients_models,device,dataloaders,writer,com,ep)
                print(loss)
                loss.backward()
                edgnn_op.step()
                edgnn_lr.step()
        model.cpu()

        pass




    writer.close()

def train_epoch_client(selected_clients_id,clients_models,optim_sched_clients,dataloaders,device,comm_dataloader):
    
    criterion = nn.CrossEntropyLoss(reduction='none') 
    loss_total=0
    diff_loss=DiffLoss()
    global Prototype_out
    Prototype_loss=PrototypeInfoNCE()
    for c in selected_clients_id:
        model=clients_models['clients'][c]
        model.to(device)
        model.train()
        for data in dataloaders['clients_train'][c]:
            x=data[0].to(device).to(torch.float32)
            y=data[1].to(device)

            optim_sched_clients['optim'][c].zero_grad() 
            output,model_fea,nomodel_fea=model(x,x)
            y=F.one_hot(y, num_classes=NUM_CLASSES)
            ce_loss = torch.sum(criterion(output, y.to(torch.float32))) 
            loss_total +=ce_loss
            se_loss = diff_loss(model_fea,nomodel_fea)
            (ce_loss+0.6*se_loss).backward()
            
            optim_sched_clients['optim'][c].step()
            optim_sched_clients['sched'][c].step()

        for (x1,x2,y) in comm_dataloader: 
            x1=x1.to(device).to(torch.float32)
            x2=x2.to(device).to(torch.float32)
            y=y.to(device)

            optim_sched_clients['optim'][c].zero_grad() 
            if c <CLIENTS:

                output,model_fea,nomodel_fea=model(x1,x1)
            else:
                output,model_fea,nomodel_fea=model(x2,x2)

            y=F.one_hot(y, num_classes=NUM_CLASSES)
            ce_loss = torch.sum(criterion(output, y.to(torch.float32))) 
            loss_total +=ce_loss
            se_loss = diff_loss(model_fea,nomodel_fea)
            if Prototype_out is None:
                Prototype_out=torch.load("/data/fusion_prototype.pt")
            pro_loss = Prototype_loss(nomodel_fea,torch.tensor(Prototype_out).to(torch.float32).detach().to(device),y)
            
            (ce_loss+0.3*se_loss+0.3*pro_loss).backward()
            
            optim_sched_clients['optim'][c].step()
            optim_sched_clients['sched'][c].step()
        model.cpu()
    return loss_total
## Main
if __name__ == '__main__':

    choose_GPU="cuda:"+ str(GPU)
    device = torch.device(choose_GPU if torch.cuda.is_available() else "cpu")
    print(device)
    print('GPU:',device)
    print('EPOCH:',EPOCH,'   COMMUNICATION:',COMMUNICATION)
    print('CLIENTS:',CLIENTS,     'RATIO:',RATIO,'      Benevolent_client:  ',Benevolent_client)
    print('BATCH:',BATCH,      )
    print('LR:',LR,'    WDECAY:',WDECAY,'     MOMENTUM:  ',MOMENTUM)
    print('no_iid_set:' , non_iid_p)
    print('DATA_SET:',DATA_SET)

    print('Preparing data......')
    datasettrain_a = DPIC_DataSet(mode='clients',multimodel='a',path=dataset_path)
    datasettrain_v = DPIC_DataSet(mode='clients',multimodel='v',path=dataset_path)

    datasettest_a = DPIC_DataSet(mode='test',multimodel='a',path=dataset_path)
    datasettest_v = DPIC_DataSet(mode='test',multimodel='v',path=dataset_path)

    datasetcommon_a = DPIC_DataSet(mode='common',multimodel='a',path=dataset_path)
    datasetcommon_v = DPIC_DataSet(mode='common',multimodel='v',path=dataset_path)

    a_label=np.array(torch.load('/data/EPIC/a_label.pt',map_location='cpu'))
    v_label=np.array(torch.load('/data/EPIC/v_label.pt',map_location='cpu'))

    a_data_splits=dirichlet_split_noniid(a_label,1,CLIENTS)
    v_data_splits=dirichlet_split_noniid(v_label,1,CLIENTS)
    global clients_data_num,total_data_num
    clients_data_num=[] 
    total_data_num=0    
    for c in range(CLIENTS*Mutimodel_num): 
        if c<CLIENTS:
            values, counts = np.unique(a_label[a_data_splits[c]], return_counts=True) 
        elif c<CLIENTS*Mutimodel_num:
            values, counts = np.unique(v_label[v_data_splits[c-10]], return_counts=True) 
        dictionary = dict(zip(values, counts))  
        ratio = np.zeros(NUM_CLASSES)   
        ratio[values] = counts
        print(' All samples : client {}, ratio {}'.format(c, ratio),'num:',np.sum(counts))
        clients_data_num.append(np.sum(counts))
        total_data_num +=np.sum(counts)

    clients_traindata_number=[] 
    clients_testdata_number =[] 
    clients_train_dataloader = []  
    clients_test_dataloader = []
    common_dataloader=[]
    common_dataloader.append(DataLoader(datasetcommon_a,batch_size=BATCH))
    common_dataloader.append(DataLoader(datasetcommon_v,batch_size=BATCH))
    
    np.random.seed(42)
    for c in range(CLIENTS*Mutimodel_num): 
        if c<CLIENTS:
            data=a_data_splits[c]
        elif c<CLIENTS *Mutimodel_num :
            data=v_data_splits[c-10]
        
        if c<10:
            clients_train_dataloader.append(DataLoader(datasettrain_a, batch_size=BATCH,sampler=SubsetSequentialSampler(data),num_workers=0,pin_memory=False,drop_last=True))
            clients_test_dataloader.append(DataLoader(datasettest_a, batch_size=BATCH,num_workers=0,pin_memory=False,drop_last=True))
        elif c<20:
            clients_train_dataloader.append(DataLoader(datasettrain_v, batch_size=BATCH,sampler=SubsetSequentialSampler(data),num_workers=0,pin_memory=False,drop_last=True))
            clients_test_dataloader.append(DataLoader(datasettest_v, batch_size=BATCH,num_workers=0,pin_memory=False,drop_last=True))
        
    dataloaders={'clients_train':clients_train_dataloader,
                 'clients_test' :clients_test_dataloader,
                 'common' : common_dataloader}
    
    
    clients_models=[] 
    random.seed(42)
    initial_modela=Attention(FEADIM,FEADIM,NUM_CLASSES,'512,256',drop_para).to('cpu')
    initial_modelv=Attention(FEADIM,FEADIM,NUM_CLASSES,'512,256',drop_para).to('cpu')
    for c in range(CLIENTS*Mutimodel_num):
        if c<10:
            clients_models.append(copy.deepcopy(initial_modela)) 
        elif c<20:
            clients_models.append(copy.deepcopy(initial_modelv)) 
    global models
    edgnn_model=EquivSetGNN(num_features_dim=96)
    models={'clients':clients_models,'edgnn':edgnn_model}
    
    train(models,dataloaders,EPOCH,device,clients_data_num)

    pass
    
