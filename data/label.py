import torch




data=torch.load('/data/EPIC/audio/audio_train_512_38000_clients.pt',map_location='cpu')
savepath='/data/EPIC/audio/epic_label.pt'
label=[]

for i in range(len(data)):
    label.append(data[i][1])

torch.save(label,savepath)

