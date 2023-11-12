import torch
from torch import nn
import torch.utils.data as Data
p=5
timecount=10000*p
Traincount=0

device = torch.device('cuda' if torch.cuda.is_available() else
'cpu')
def corr2d(X, K):
   h, w = K.shape
   Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
   for i in range(Y.shape[0]):
       for j in range(Y.shape[1]):
           Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
   return Y
class Conv2D(nn.Module):
   def __init__(self, kernel_size):
       super(Conv2D, self).__init__()
       self.weight = nn.Parameter(torch.randn(kernel_size))
       self.bias = nn.Parameter(torch.randn(1))
   def forward(self, x):
       return corr2d(x, self.weight) + self.bias
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量⼤⼩和通道数（“多输⼊通道和多输出通道”⼀节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:]) # 排除不关⼼的前两维：批量和通
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
trianbatch=3
lr, num_epochs = 0.001 *trianbatch, 200

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
       self.fc = nn.Sequential(
           nn.Linear(50*5*channel3, channel3),
           nn.Sigmoid(),
           nn.Linear(channel3, 1),

       )
   def forward(self, img):
       feature = self.conv(img)
       output = self.fc(feature.view(img.shape[0], -1))
       return output
net = MyNet()
lossxlist=[i for i in range(num_epochs)]
lossylist=[i for i in range(num_epochs)]



def Train(ltlist,llist,ptlist,plist,ptf,ltf):
    global lossxlist
    global lossylist
    global net
    loss = nn.MSELoss()
    x = [i for i in range(int(abs(ptf - ltf)), timecount)]
    #random.shuffle(x)
    nplist = []
    nllist = []
    flag = 0
    for num in x:
        for lt in range(len(ltlist)):
            if (ltlist[lt] > (num/p)) and lt/10!=0:
                for pt in range(len(ptlist)):
                    if (ptlist[pt] + ptf - ltf+delay > num/p):
                        flag = 1
                        nplist.append(plist[pt])
                        nllist.append([llist[lt]])
                        break

                break
    if (flag == 0):
        x = [i for i in range(int(abs(ptf - ltf + 12 * 60 * 60)), timecount)]
        #random.shuffle(x)
        for num in x:
            for lt in range(len(ltlist)):
                if (ltlist[lt] > (num/p)) and lt/10!=0:
                    for pt in range(len(ptlist)):
                        if (ptlist[pt] + ptf - ltf + 12 * 60 * 60+delay > num/p):
                            nplist.append(plist[pt])
                            nllist.append([llist[lt]])
                            break
                    break
    y = torch.tensor(nplist)
    x = torch.tensor(nllist)
    torch_dataset = Data.TensorDataset(x, y)
    loader=Data.DataLoader(
        dataset=torch_dataset,
        batch_size=trianbatch,
        shuffle=True,
        num_workers=2,
    )
    for epoch in range(1, num_epochs + 1):
        global Traincount
        for i,(bx,by) in enumerate(loader) :
            bx,by=bx.to(device),by.to(device).unsqueeze(-1)
            r = lr / (5 + epoch)
            optimizer = torch.optim.Adam(net.parameters(), lr=r)
            y_hat = net(bx)
            l = loss(y_hat, by).mean()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            print("epoch:{},loss:{}".format(epoch,l))
        Traincount = epoch / num_epochs * 100

