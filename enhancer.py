import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from util.config import *
class Common_DataSet(Dataset):
    def __init__(self, mode='common',path=''):   # 
        super(Common_DataSet, self).__init__()
       
       
           
        self.file_path=path
        if mode=='common':
            save_path1='/data/EPIC/audio/a_common.pt'
            save_path2='/data/EPIC/video/v_common.pt'
        else:
            save_path1='/data/EPIC/audio/a_test.pt'
            save_path2='/data/EPIC/video/v_test.pt'

        self.data1=torch.load(save_path1,map_location='cpu')
        self.data2=torch.load(save_path2,map_location='cpu')

        self.leng=len(self.data1)
        
    def __len__(self):
        return self.leng

    def __getitem__(self, idx):
        mod1,y1=self.data1[idx]   #[feature,label]
        mod2,y2=self.data2[idx]
        if y1!=y2:
            print("Public dataset label error")
        return [mod1,mod2,y1]



Prototype_out=None
class PrototypeLoss(nn.Module):
    def __init__(self, num_classes, lambda_reg=0.15):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
    
    def forward(self, feature_prototypes,prototypes, labels):
        batch_prototypes = prototypes[labels]
        
        l2_dist = torch.norm(feature_prototypes - batch_prototypes, p=2, dim=1)
        
        prototype_loss = torch.mean(l2_dist)
        
        return self.lambda_reg * prototype_loss

class Multimodal_enhancer(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, num_classes, prototype_dim=64):
        super().__init__()
        self.mod1_branch = nn.Sequential(
            nn.Linear(mod1_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.mod2_branch = nn.Sequential(
            nn.Linear(mod2_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )
        
        self.prototype_layer = nn.Sequential(
            nn.Linear(256, prototype_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(prototype_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.soft=nn.Softmax(dim=1)
    def forward(self, x1, x2):
    
        mod1_feat = self.mod1_branch(x1)
        mod2_feat = self.mod2_branch(x2)
        combined = torch.cat((mod1_feat, mod2_feat), dim=1)
        attn_weights = self.attention(combined)

        fused_feat = attn_weights[:, 0:1] * mod1_feat + attn_weights[:, 1:2] * mod2_feat

        feature_prototype = self.prototype_layer(fused_feat)
        logits = self.classifier(feature_prototype)

        return self.soft(logits), feature_prototype, mod1_feat,mod2_feat

def Prototype_Enhancer(pro_model=None,device=None,mutimodal_prototype_clients=None,dataloaders=None):
    global Prototype_out  
    if Prototype_out ==None:
        Prototype_out=torch.load("/data/fusion_prototype.pt")
    
    def cosine_similarity(a, b):
        dot_product = np.sum(a * b, axis=1)  
        norm_a = np.linalg.norm(a, axis=1)   
        norm_b = np.linalg.norm(b, axis=1)   
        similarity = dot_product / (norm_a * norm_b)
        return np.exp(similarity)
    modal_num=len(mutimodal_prototype_clients)
    id_num=len(mutimodal_prototype_clients[0])
    class_num=len(mutimodal_prototype_clients[0][0])
    similarity_matrix = []
    
    
    for j in range(modal_num):
        similarity_matrix.append(np.zeros((id_num,class_num)))
        for i in range(id_num):
            similarity_matrix[j][i] = cosine_similarity(mutimodal_prototype_clients[j][i], Prototype_out)
    exp_similarity_matrix = np.exp(similarity_matrix - np.max(similarity_matrix, axis=1, keepdims=True))  
    norm_similarity_matrix = exp_similarity_matrix / np.sum(exp_similarity_matrix, axis=1, keepdims=True)
    
    aggregated_prototypes = np.einsum('ijk,ijkm->ikm', norm_similarity_matrix, mutimodal_prototype_clients)
    loss_Prototype=PrototypeLoss(NUM_CLASSES) #[2,101,256]
    criterion = nn.CrossEntropyLoss(reduction='mean') 
    learning_rate = 0.005
    optimizer = optim.Adam(pro_model.parameters(), lr=learning_rate)
    pro_model.to(device).train()
    fusion_prototype=[[] for _ in range(NUM_CLASSES)]
    round=50
    
    for ep in range(round):
        for (x1,x2,y) in dataloaders:
            x1=x1.to(device).to(torch.float32)
            x2=x2.to(device).to(torch.float32)
            y=y.to(device)
            pre,fusion_prototype_feat,mod1_feat,mod2_feat = pro_model(x1, x2)

            loss = criterion(pre, y)  
            loss2=loss_Prototype(mod1_feat,torch.tensor(aggregated_prototypes[0]).to(device),y) + loss_Prototype(mod2_feat,torch.tensor(aggregated_prototypes[1]).to(device),y)
            optimizer.zero_grad()
            (loss+loss2).backward()
            optimizer.step()
            if ep==round-1:
                for i in range(y.size(0)): 
                    label=y[i].item()
                    fusion_prototype[label].append(fusion_prototype_feat[i].detach().cpu().numpy().tolist())

    
    Prototype_out=comp_mean(fusion_prototype)



    pass

def comp_mean(fusion_prototype):
    mean_values = np.zeros((NUM_CLASSES, 256))
    valid_counts = np.zeros((NUM_CLASSES, 1))

    for i in range(NUM_CLASSES):
        for sublist in fusion_prototype[i]:
            if len(sublist) > 0:  
                mean_values[i] += sublist
                valid_counts[i] += 1

    fusion_prototype = np.divide(mean_values, valid_counts, out=np.zeros_like(mean_values), where=valid_counts != 0)

    return fusion_prototype
## Main
if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss(reduction='mean') 
    datasetcommon = Common_DataSet(mode='common',path=None)
    common_dataloader_mutimodal=DataLoader(datasetcommon,batch_size=BATCH)

    modal_enhancer=Multimodal_enhancer(mod1_dim=2048, mod2_dim=2048, num_classes=NUM_CLASSES, prototype_dim=256)
    learning_rate = 0.005
    optimizer = optim.Adam(modal_enhancer.parameters(), lr=learning_rate)
    device = torch.device(4)
    modal_enhancer.to(device).train()
    fusion_prototype=[[] for _ in range(NUM_CLASSES)]
    mod1_prototype=[[] for _ in range(NUM_CLASSES)]
    mod2_prototype=[[] for _ in range(NUM_CLASSES)]
    round=2
    for ep in range(round):

        for (x1,x2,y) in common_dataloader_mutimodal:
            x1=x1.to(device).to(torch.float32)
            x2=x2.to(device).to(torch.float32)
            y=y.to(device)
            
            
            pre,fusion_prototype_feat,mod1_feat,mod2_feat = modal_enhancer(x1, x2)

            loss = criterion(pre, y)  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep==round-1:
                for i in range(y.size(0)): 
                    label=y[i].item()
                    fusion_prototype[label].append(fusion_prototype_feat[i].detach().cpu().numpy().tolist())
                    mod1_prototype[label].append(mod1_feat[i].detach().cpu().numpy().tolist())
                    mod2_prototype[label].append(mod2_feat[i].detach().cpu().numpy().tolist())



        #test
        datasettest = Common_DataSet(mode='test',path=None)
        test_dataloader=DataLoader(datasettest,batch_size=BATCH)
        modal_enhancer.eval()  
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  
            for (x1,x2,y) in test_dataloader:
                x1=x1.to(device).to(torch.float32)
                x2=x2.to(device).to(torch.float32)
                y=y.to(device)

                
                pre, fusion_prototype22, mod1_feat22, mod2_feat22 = modal_enhancer(x1, x2)

                
                loss = criterion(pre, y)
                total_loss += loss.item()

                
                _, predicted = torch.max(pre, 1)
                correct_predictions += (predicted == y).sum().item()
                total_samples += y.size(0)

        
        average_loss = total_loss / len(test_dataloader)
        accuracy = correct_predictions / total_samples

        print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    
    fusion_prototype=comp_mean(fusion_prototype)
    torch.save(fusion_prototype,'/data/fusion_prototype.pt')

