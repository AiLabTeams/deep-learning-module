#import torch
#from torch import nn, optim
#import torch.utils.data as Data
#import random
#import math
#p=5
#timecount=10000*p
#Traincount=0
#
#device = torch.device('cuda' if torch.cuda.is_available() else
#'cpu')
#
#class tdataset(Data.Dataset):
#    def _init_(self,data_tensor,target_tensor):
#        self.data_tensor=data_tensor
#        self.target_tensor=target_tensor
#    def _getitem_(self,index):
#        return self.data_tensor[index],self.target_tensor[index]
#    def _len_(self):
#        return self.data_tensor.size(0)
#
#delay=5
#input_size=45
#hidden_size=45
#num_layer=3
#seq_len=20
#batch=1
#num_direction=1
#myin=torch.randn(batch,seq_len,input_size)
#h_0=torch.randn(num_layer*num_direction,batch,hidden_size)
#lr, num_epochs = 0.0001 *batch, 15
#class MyNet(nn.Module):
#   def __init__(self):
#       super(MyNet, self).__init__()
#
#       self.gru=nn.GRU(input_size, hidden_size, num_layer,batch_first=True,bidirectional=False)
#       for p in self.gru.parameters():
#           nn.init.normal_(p,mean=0,std=0.5)
#       self.fc = nn.Sequential(
#           nn.Linear(seq_len*hidden_size, hidden_size),
#           nn.ReLU(),
#           nn.Linear(hidden_size, 1),
#       )
#   def forward(self, img,h0):
#       feature,h1 = self.gru(img,h0)
#       output = self.fc(feature.view(img.shape[0], -1))
#       return output,h1
#net = MyNet()
#def Train(ltlist,llist,ptlist,plist,ptf,ltf):
#    global net
#    loss = nn.MSELoss()
#    x = [i for i in range(int(abs(ptf - ltf))+20, timecount)]
#    #random.shuffle(x)
#    nplist = []
#    nllist = []
#    flag = 0
#    for num in x:
#        for lt in range(len(ltlist)):
#            if (ltlist[lt] > (num/p)):
#                for pt in range(len(ptlist)):
#                    if (ptlist[pt] + ptf - ltf +delay> num/p):
#                        flag = 1
#                        nplist.append(plist[pt])
#                        nllist.append([llist[lt-19+i] for i in range(20)])
#                        break
#
#                break
#    if (flag == 0):
#        x = [i for i in range(int(abs(ptf - ltf + 12 * 60 * 60))+20, timecount)]
#        #random.shuffle(x)
#        for num in x:
#            for lt in range(len(ltlist)):
#                if (ltlist[lt] > (num/p)):
#                    for pt in range(len(ptlist)):
#                        if (ptlist[pt] + ptf - ltf + 12 * 60 * 60+delay > num/p):
#                            nplist.append(plist[pt])
#                            nllist.append([llist[lt-19+i] for i in range(20)])
#                            break
#                    break
#    y = torch.tensor(nplist)
#    x = torch.tensor(nllist)
#    torch_dataset = Data.TensorDataset(x, y)
#    loader=Data.DataLoader(
#        dataset=torch_dataset,
#        batch_size=batch,
#        shuffle=True,
#        num_workers=2,
#    )
#    for epoch in range(1, num_epochs + 1):
#        global Traincount
#        for i,(bx,by) in enumerate(loader) :
#            bx,by=bx.to(device),by.to(device).unsqueeze(-1)
#            r = lr / (1 + epoch)
#            optimizer = torch.optim.SGD(net.parameters(), lr=r)
#            y_hat,h = net(bx,h_0)
#            l = loss(y_hat, by).mean()
#            optimizer.zero_grad()
#            l.backward()
#            optimizer.step()
#            print("epoch:{},loss:{}".format(epoch,l))
#
#        Traincount = epoch / num_epochs * 100
import torch
from torch import nn
import torch.utils.data as Data
p=5
timecount=10000*p
Traincount=0

device = torch.device('cuda' if torch.cuda.is_available() else
'cpu')

class tdataset(Data.Dataset):
    def _init_(self,data_tensor,target_tensor):
        self.data_tensor=data_tensor
        self.target_tensor=target_tensor
    def _getitem_(self,index):
        return self.data_tensor[index],self.target_tensor[index]
    def _len_(self):
        return self.data_tensor.size(0)

delay=1
channel1=5
channel2=25
channel3=20
input_size=channel3
hidden_size=channel3
num_layer=2
seq_len=5
batch=1
num_direction=1
myin=torch.randn(batch,seq_len,input_size)
h_0=torch.randn(num_layer*num_direction,batch,hidden_size)
lr, num_epochs = 0.0001 *batch, 100
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(1, channel1, 3, 2),  # in_channels, out_channels,kernel_size
           nn.BatchNorm2d(channel1),
           nn.Sigmoid(),
           nn.Conv2d(channel1, channel2, 2, 2),
           nn.BatchNorm2d(channel2),
           nn.Sigmoid(),
           nn.Conv2d(channel2, channel3, 4, 1),
           nn.BatchNorm2d(channel3),
           nn.Sigmoid(),
           nn.Conv2d(channel3, channel3, 4, 1),
           nn.BatchNorm2d(channel3),
           nn.Sigmoid(),

       )
       self.gru=nn.GRU(input_size, hidden_size, num_layer,batch_first=True,bidirectional=False)
       #for p in self.gru.parameters():
       #    nn.init.normal_(p,mean=0,std=0.5)
       self.fcc = nn.Sequential(
           nn.Linear(5 * 5 * channel3, channel3),
       )
       self.fcg = nn.Sequential(
           nn.Linear(seq_len*hidden_size, 1),
       )
   def forward(self, imgs,h0):
       i0 = self.conv(imgs[0])
       i1 = self.conv(imgs[1])
       i2 = self.conv(imgs[2])
       i3 = self.conv(imgs[3])
       i4 = self.conv(imgs[4])
       ic0= self.fcc(i0.view(imgs[0].shape[0], -1))
       ic1 = self.fcc(i1.view(imgs[1].shape[0], -1))
       ic2 = self.fcc(i2.view(imgs[2].shape[0], -1))
       ic3 = self.fcc(i3.view(imgs[3].shape[0], -1))
       ic4 = self.fcc(i4.view(imgs[4].shape[0], -1))
       im=torch.tensor([ic0,ic1,ic2,ic3,ic4])
       feature,h1 = self.gru(im,h0)
       output = self.fcg(feature.view(im.shape[0], -1))
       return output,h1
net = MyNet()
def Train(ltlist,llist,ptlist,plist,ptf,ltf):
    global net
    loss = nn.MSELoss()
    x = [i for i in range(int(abs(ptf - ltf)), timecount)]
    # random.shuffle(x)
    nplist = []
    nllist = []
    flag = 0
    for num in x:
        for lt in range(len(ltlist)-5):
            if (ltlist[lt] > (num / p)) and lt / 10 != 0:
                for pt in range(len(ptlist)):
                    if (ptlist[pt] + ptf - ltf + delay > num / p):
                        flag = 1
                        nplist.append(plist[pt])
                        nllist.append([[llist[lt]], [llist[lt + 1]], [llist[lt + 2]], [llist[lt + 3]], [llist[lt + 4]]])
                        break

                break
    if (flag == 0):
        x = [i for i in range(int(abs(ptf - ltf + 12 * 60 * 60)), timecount)]
        # random.shuffle(x)
        for num in x:
            for lt in range(len(ltlist) - 5):
                if (ltlist[lt] > (num / p)) and lt / 10 != 0:
                    for pt in range(len(ptlist)):
                        if (ptlist[pt] + ptf - ltf + 12 * 60 * 60 + delay > num / p):
                            nplist.append(plist[pt])
                            nllist.append(
                                [[llist[lt]], [llist[lt + 1]], [llist[lt + 2]], [llist[lt + 3]], [llist[lt + 4]]])
                            break
                    break
    y = torch.tensor(nplist)
    x = torch.tensor(nllist)
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )
    for epoch in range(1, num_epochs + 1):
        global Traincount
        for i,(bx,by) in enumerate(loader) :
            bx,by=bx.to(device),by.to(device).unsqueeze(-1)
            r = lr / (1 + epoch)
            optimizer = torch.optim.SGD(net.parameters(), lr=r)
            y_hat,h = net(bx,h_0)
            l = loss(y_hat, by).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            print("epoch:{},loss:{}".format(epoch,l))

        Traincount = epoch / num_epochs * 100
