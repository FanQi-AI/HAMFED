''' Configuration File.
'''


# setting
EPOCH=2 
COMMUNICATION =200 
RATIO=0.8  
CLIENTS=10  
NUM_CLASSES = 101 
GPU= 5 
Benevolent_client=0.8  
Mutimodel_num=2 
#clients training
LR = 0.001 
# NUM_TRAIN =  280000  
WDECAY = 5e-4 
BATCH = 32 
MOMENTUM = 0.9 
MILESTONES =[80]
DATA_SET='EPIC' 

EDGNN_LR=0.03
EDGNN_EPOCH=20
FEADIM=1024
drop_para=0.3


#non-iid seting
non_iid_p=0.7   
non_iid_seed=13    
non_iid_alpha_dirichlet=10   

#数据集地址
dataset_path = r'/data/EPIC/feature'
