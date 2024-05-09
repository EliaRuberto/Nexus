import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#rete NEURALE 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optin

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True,transform=transform)
#insegnare all AI
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True, num_workers=2)
#ITERARE I DATI INSEGNATI
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
#TESTARE I DATI IN POSSESSO
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,
                                         shuffle=False,num_workers=2)
#ITERARE I DATI TESTATI E SALVARLI
classes = (' aereo ',' macchina ',' uccello ',' gatto ',' cervo ',' cane ',' rana ',' cavallo ',' nave ',' camion ')

#IMPORTARE LE IMMAGINI
def imshow(img):
    img=img / 2+0.5
    npimg= img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
#PRENDERE IMMAGINGI RANDAOM DAL DATALIZER

dataiter= iter(trainloader)
images,labels = next(dataiter)

#MOSTRARE PRIMA LE IMMAGINI E POI DARE UNA CLASSE
imshow(torchvision.utils.make_grid(images))

print(''.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

#IMPOSTARE LA RETE NEURALE PER APPRENDIMENTO
class Net(nn.Module):
    def __init__(self): #TUTTI I LAYER PER LA RETE NEURALE
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1= nn.Linear(16*5*5,out_features=120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Non applicare ReLU a self.fc2
        x = self.fc3(x)
        return x



    
net=Net()

criterion = nn.CrossEntropyLoss()
optimizer = optin.SGD(net.parameters(),lr=0.001,momentum=0.9) #lr = leareinbg rate porcDOSFIAOSDFIUODS

# 2 cicli
for epoch in range(2): #CICLARE IL DATASERT
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): #ITERA PER OGNI BATCH
        inputs,labels =data   # CARICARE GLI INPUT CON BASE DI APPARTENENZA
        optimizer.zero_grad() # AZZERARE I GRADE PER NON SOMMARLI
        outputs= net(inputs)
        loss=criterion(outputs,labels)# CALCOLA IL GRADIENTE PER LA RISPOSTA
        loss.backward()
        optimizer.step() #AGGIORNARE I PARAMETRI

        running_loss += loss.item()
        if i %2000 == 1999:
            print(f'[{epoch + 1},{i + 1:5d}] loss: {running_loss /2000 :.3f}')# BATCH PER ERRORE ATTUALE
            running_loss=0
print('allenamento finito andate in pace')

#SALVARE 
PATH= './cifar_net.pth'
torch.save(net.state_dict(),PATH)
#carica
net=Net()
net.load_state_dict(torch.load(PATH))
#printa immagini

imshow(torchvision.utils.make_grid(images))
print('La vera classe: ',''.join(f'{classes[labels[j]]:5s}'for j in range(4)))
outputs= net(images)
_, predicted = torch.max(outputs, 1)
print('Classe predetta: ',''.join(f'{classes[labels[j]]:5s}'
                             for j in range(4)))


correct_pred={classname:0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels =data
        outputs = net(images)
        _, predictions = torch.max(outputs,1)
        for label,predictions in zip(labels,predictions):
            if label == predictions:
                total_pred[classes[label]] +=1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count)/total_pred[classname]
    print(f'accuratezza per classe: {classname:5s} e {accuracy:1f}%')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)
    inputs, labels = data[0].to(device), data[1].to(device)
device = torch.device('cuda:0' if torch.cuda.is_available()else'cpu')
print(device)
net.to(device)
inputs, labels = data[0].to(device), data[1].to(device)
    
