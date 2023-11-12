import torch
from torch import nn
import torch.utils.data as Data
import random
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
channel2=20
channel3=25
batch=1
num_direction=1
lr, num_epochs = 0.0008 *batch, 200
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.convs = nn.Sequential(
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
       self.convd = nn.Sequential(
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
       self.fcs = nn.Sequential(
           nn.Linear(50 * 5 * channel3, 1),
       )
       self.fcd = nn.Sequential(
           nn.Linear(50 * 5 * channel3, 1),
       )

       self.fc = nn.Sequential(
           nn.Linear(2, 1)
       )
   def forward(self, imgs,imgd):
       features = self.convs(imgs)
       featured = self.convd(imgd)
       outputs = self.fcs(features.view(imgs.shape[0], -1))
       outputd = self.fcd(featured.view(imgs.shape[0], -1))
       output = self.fc(torch.cat((outputs,outputd),1))
       return output
net = MyNet()
nplist = []
nllist1 = []
nllist2 = []
Datacount=0
def Dataget(ltlist,llist1,llist2,ptlist,plist,ptf,ltf):
    x = [i for i in range(int(abs(ptf - ltf)), timecount)]
    flag = 0
    global Datacount
    for num in x:
        Datacount=100*num/x[len(x)-1]
        for lt in range(len(ltlist)):
            if (ltlist[lt] > (num / p)) and lt / 10 != 0:
                for pt in range(len(ptlist)):
                    if (ptlist[pt] + ptf - ltf + delay > num / p):
                        flag = 1
                        nplist.append(plist[pt])
                        nllist1.append([[llist1[lt]]])
                        nllist2.append([[llist2[lt]]])
                        break
                break
    if (flag == 0):
        x = [i for i in range(int(abs(ptf - ltf + 12 * 60 * 60)), timecount)]
        for num in x:
            Datacount = 100 * num / x[len(x) - 1]
            for lt in range(len(ltlist)):
                if (ltlist[lt] > (num / p)) and lt / 10 != 0:
                    for pt in range(len(ptlist)):
                        if (ptlist[pt] + ptf - ltf + 12 * 60 * 60 + delay > num / p):
                            nplist.append(plist[pt])
                            nllist1.append([[llist1[lt]]])
                            nllist2.append([[llist2[lt]]])
                            break
                    break
def Train(ltlist,llist1,llist2,ptlist,plist,ptf,ltf):
    global net
    loss = nn.MSELoss()
    nptensor = torch.tensor(nplist)
    nltensor1 = torch.tensor(nllist1)
    nltensor2 = torch.tensor(nllist2)
    x = [i for i in range(len(nplist))]
    for epoch in range(1, num_epochs + 1):

        random.shuffle(x)
        global Traincount
        for i in x :
            np,nl1,nl2=nptensor[i].to(device).unsqueeze(-1),nltensor1[i].to(device),nltensor2[i].to(device)
            r = lr / (5 + epoch)
            optimizer = torch.optim.Adam(net.parameters(), lr=r)
            y_hat= net(nl1,nl2)
            l = loss(y_hat, np).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            print("epoch:{},loss:{}".format(epoch,l))

        Traincount = epoch / num_epochs * 100