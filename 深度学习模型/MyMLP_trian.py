from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import WavePackage
import PressPackage
import math
from scipy.optimize import curve_fit
from scipy.fftpack import fft,ifft
import random
import WavePackage
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torch import nn
mod = "mlp202301118"
LightWaveAmplitudeListsRaw = []
LightWaveAmplitudeListsFilter = []
LightWaveAmplitudeLists = []
LightWaveLengthList = []
LightWaveTimeList = []
LightWaveTimeSecondList = []
PressTimeList = []
PressAmplitudeList = []
NewPressAmplitudeList = []
PressAmplitudelogList = []
lightfilecount=0
PressTimeToFix=0
LightTimeToFix=0


Allfftlist=[]
AllPressTimeToFix=[]
AllLightTimeToFix=[]
AllPressTimeList = []
AllPressAmplitudelogList = []
AllLightWaveTimeSecondList=[]
AllLightWaveAmplitudeLists=[]

fftabslistlist = []
fftanglelist = []

Allxlist=[]
Allylist=[]
xtrainlist=[]
ytrianlist=[]
yprelist=[]
xprelist=[]
timedis=300

batch=1
lr, num_epochs = 0.01 , 200
channelmut=5
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.mlp=nn.Sequential(
           #nn.Sigmoid(),
           nn.Linear(len(xtrainlist[0][0]), len(xtrainlist[0][0]) * channelmut),
           nn.Sigmoid(),
           nn.Linear(len(xtrainlist[0][0]) * channelmut, len(xtrainlist[0][0]) * channelmut),
           nn.Sigmoid(),
           nn.Linear(len(xtrainlist[0][0]) * channelmut, len(xtrainlist[0][0]) * channelmut),
           nn.Sigmoid(),
           nn.Linear(len(xtrainlist[0][0])*channelmut, 1),
       )
   def forward(self, myin):
       output = self.mlp(myin.to(torch.float32))
       return output


def Import_light_file(fpath):
    global LightWaveAmplitudeListsRaw
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    global LightWaveLengthList
    global LightWaveTimeList
    global LightWaveTimeSecondList
    global lightfilecount
    global LightTimeToFix
    try:
        t = WavePackage.Wave(fpath)

        if lightfilecount == 0:
            LightTimeStart = t.TimeStart
            LightTimeToFix = t.IntWaveTimeList[0] % 100000000000 / 1000
            LightWaveLengthList = t.FloatWaveLengthList
            LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists

            LightWaveTimeList += t.IntWaveTimeList
            LightWaveTimeSecondList += t.FloatWaveTimeSecondList
            lightfilecount = lightfilecount + 1
            LightTimeEnd = t.TimeEnd
        else:
            LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists
            LightWaveTimeList += t.IntWaveTimeList
            temp = len(LightWaveTimeSecondList)
            LightWaveTimeSecondList += [i + LightWaveTimeSecondList[1] - LightWaveTimeSecondList[0] +
                                             LightWaveTimeSecondList[temp - 1]
                                             for i in t.FloatWaveTimeSecondList]
            lightfilecount = lightfilecount + 1
            LightTimeEnd = t.TimeEnd
        LightWaveAmplitudeLists = LightWaveAmplitudeListsRaw
        print('光谱数据读入成功')
        print('当前载入光谱文件数量: ' + str(lightfilecount))
    except:
        print('光谱数据读入失败')
    pass
def LightWavefilter():
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    LightWaveAmplitudeListsFilter=LightWaveAmplitudeLists.copy()
    templist=[0 for i in range(len(LightWaveAmplitudeLists))]
    for i in range(len(LightWaveAmplitudeLists[0])):
        for iii in range(len(LightWaveAmplitudeLists)):
            templist[iii]=LightWaveAmplitudeLists[iii][i]
        templist=savgol_filter(templist, 5, 1)
        for iii in range(len(LightWaveAmplitudeLists)):
               LightWaveAmplitudeListsFilter[iii][i]=templist[iii]
    LightWaveAmplitudeLists=LightWaveAmplitudeListsFilter.copy()

def Import_press_file(fpath):
    global PressTimeList
    global PressAmplitudeList
    global NewPressAmplitudeList
    global PressAmplitudelogList
    global PressTimeToFix
    try:
        t = PressPackage.Press(fpath)
        PressTimeStart = t.TimeStart
        PressTimeToFix = t.IntPressTimeList[0] % 100000000000 / 1000
        PressTimeList = t.IntPressTimeSecondList

        PressAmplitudeList = t.IntPressDataList
        PressTimeEnd = t.TimeEnd
        PressMove = 0.015
        temp = 0
        pmin = min(PressAmplitudeList)
        PressAmplitudeList=[i-pmin for i in PressAmplitudeList]
        for i in range(len(PressAmplitudeList)):
            temp = temp + PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] > 10:
                temp = temp - PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] < -10:
                temp = 0
            NewPressAmplitudeList.append(max(PressAmplitudeList[i] - temp, 5))
            #NewPressAmplitudeList.append(min(PressAmplitudeList[i], PressAmplitudeList[len(PressAmplitudeList)-1]*1.01))
        PressAmplitudelogList = [math.log(max(5, i), 10) for i in NewPressAmplitudeList]
        print('气压数据读入成功')
    except:
        print('气压数据读入失败')

def fftfuc():
    for i in range(len(LightWaveAmplitudeLists)):
        ffty = fft(LightWaveAmplitudeLists[i])
        fftyabs = [abs(i) for i in ffty]
        fftyangle = [i.imag for i in ffty]
        fftabslistlist.append(fftyabs[0:5])
        fftanglelist.append(fftyangle[0:5])
    pass
def SGDfuc():
    for i in range(len(LightWaveAmplitudeLists)):
        LightWaveAmplitudeLists[i]=savgol_filter(LightWaveAmplitudeLists[i],len(LightWaveAmplitudeLists[i])//2,5)
    pass
def Dataget():
    global Allxlist
    global Allylist
    xlist=[]
    ylist=[]
    if abs(PressTimeToFix - LightTimeToFix) > 36000:
        temp = PressTimeToFix - LightTimeToFix + 12 * 60 * 60
    else:
        temp = PressTimeToFix - LightTimeToFix
    for pt in range(timedis*6//10+50,timedis+50):
        for lt in range(5, len(LightWaveTimeSecondList)):
            if LightWaveTimeSecondList[lt] >= PressTimeList[pt] + temp:
                templist = []
                # templist.extend(fftabslistlist[lt].copy())
                # templist.extend(fftanglelist[lt].copy())
                # templist.extend(LightWaveAmplitudeLists[lt][::15])
                templist.extend(LightWaveAmplitudeLists[lt][::15])
                xlist.append(templist.copy())
                ylist.append(PressAmplitudelogList[pt])
                break
    for pt in range(len(PressTimeList)-timedis, len(PressTimeList)):
        for lt in range(5, len(LightWaveTimeSecondList)):
            if LightWaveTimeSecondList[lt] >= PressTimeList[pt] + temp:
                templist = []
                # templist.extend(fftabslistlist[lt].copy())
                # templist.extend(fftanglelist[lt].copy())
                # templist.extend(LightWaveAmplitudeLists[lt][::15])
                templist.extend(LightWaveAmplitudeLists[lt][::15])
                xlist.append(templist.copy())
                ylist.append(PressAmplitudelogList[pt])
                break
    Allxlist.append(xlist.copy())
    Allylist.append(ylist.copy())
    print('dataget 成功')
def zero():
    global LightWaveAmplitudeLists
    templen=20
    tempexlist=[0 for i in range(len(LightWaveAmplitudeLists[0]))]
    for ii in range(len(LightWaveAmplitudeLists[0])):
        tempsum=0.0
        for iii in range(templen):
            tempsum=LightWaveAmplitudeLists[iii+10][ii]+tempsum
        tempexlist[ii]=tempsum/templen
    for i in range(0,len(LightWaveAmplitudeLists)):
        for ii in range(len(LightWaveAmplitudeLists[0])):
            LightWaveAmplitudeLists[i][ii] = LightWaveAmplitudeLists[i][ii] - tempexlist[ii]
def next():
    global LightWaveAmplitudeListsRaw
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    global LightWaveLengthList
    global LightWaveTimeList
    global LightWaveTimeSecondList
    global lightfilecount
    global PressTimeList
    global PressAmplitudeList
    global NewPressAmplitudeList
    global PressAmplitudelogList
    global AllPressTimeToFix
    global AllLightTimeToFix
    global AllPressTimeList
    global AllPressAmplitudelogList
    global AllLightWaveTimeSecondList
    global fftabslistlist
    global fftanglelist
    AllPressTimeToFix.append(PressTimeToFix)
    AllLightTimeToFix.append(LightTimeToFix)
    AllPressTimeList.append(PressTimeList.copy())
    AllPressAmplitudelogList.append(PressAmplitudelogList.copy())
    AllLightWaveTimeSecondList.append(LightWaveTimeSecondList)
    LightWaveAmplitudeListsRaw = []
    LightWaveAmplitudeListsFilter = []
    LightWaveAmplitudeLists = []
    LightWaveLengthList = []
    LightWaveTimeList = []
    LightWaveTimeSecondList = []
    PressTimeList = []
    PressAmplitudeList = []
    NewPressAmplitudeList = []
    PressAmplitudelogList = []
    fftabslistlist = []
    fftanglelist = []
    lightfilecount = 0
    print('清除缓存')
def loaddata(lpaths,ppath):
    for i in range(len(lpaths)):
        Import_light_file('mydata/'+lpaths[i]+'.txt')
    Import_press_file('mydata/'+ppath+'.csv')
    zero()
    fftfuc()
    SGDfuc()
    Dataget()
    next()

loaddata(['P203160901_2022-11-10 15-21-02 904','P203160901_2022-11-10 15-29-38 926'],'20221109-2432pd10')#1
loaddata(['P203160901_2022-11-10 15-43-34 744','P203160901_2022-11-10 15-52-10 297'],'20221109-2661pd10')#2
loaddata(['P203160901_2022-11-09 20-51-08 739','P203160901_2022-11-09 20-59-46 282'],'20221109-2847pd10')#3
loaddata(['P203160901_2022-11-09 16-40-45 813','P203160901_2022-11-09 16-49-22 527'],'20221109-3047pd10')#4
loaddata(['P203160901_2022-11-10 21-49-48 028','P203160901_2022-11-10 21-58-25 088'],'20221109-3152pd10')#5
loaddata(['P203160901_2022-11-09 16-20-32 054','P203160901_2022-11-09 16-29-08 129'],'20221109-3201pd10')#1
loaddata(['P203160901_2022-11-11 16-01-06 530','P203160901_2022-11-11 16-09-42 663'],'20221109-3226pd10')#2
loaddata(['P203160901_2022-11-10 15-00-01 436','P203160901_2022-11-10 15-08-39 531'],'20221109-3273pd10')#3
loaddata(['P203160901_2022-11-10 13-29-35 324','P203160901_2022-11-10 13-38-11 975'],'20221109-3493pd10')#4
loaddata(['P203160901_2022-11-10 16-03-56 353','P203160901_2022-11-10 16-12-32 459'],'20221109-3685pd10')#5
#loaddata(['P203160901_2022-11-09 19-46-56 710','P203160901_2022-11-09 19-55-32 268'],'20221109-3794pd10')#1
loaddata(['P203160901_2022-11-11 14-14-44 805','P203160901_2022-11-11 14-23-20 424'],'20221109-3833pd10')#2
loaddata(['P203160901_2022-11-11 15-38-49 488','P203160901_2022-11-11 15-47-25 551'],'20221109-4218pd10')#3

#loaddata(['P203160901_2022-11-11 15-38-49 488','P203160901_2022-11-11 15-47-25 551'],'20221109-4218pd10')#3
#loaddata(['P203160901_2022-11-10 14-17-28 710','P203160901_2022-11-10 14-26-04 251'],'20221109-4395pd10')#4
loaddata(['P203160901_2022-11-09 21-12-55 253','P203160901_2022-11-09 21-21-31 775'],'20221109-4577pd10')#5
loaddata(['P203160901_2022-11-10 22-29-18 171','P203160901_2022-11-10 22-37-55 972'],'20221109-4595pd10')#1
loaddata(['P203160901_2022-11-10 17-48-11 735','P203160901_2022-11-10 17-56-47 297'],'20221109-4883pd10')#2
loaddata(['P203160901_2022-11-10 17-25-41 128','P203160901_2022-11-10 17-34-17 216'],'20221109-5112pd10')#3
loaddata(['P203160901_2022-11-10 17-04-44 910','P203160901_2022-11-10 17-13-20 460'],'20221109-5687pd10')#4
loaddata(['P203160901_2022-11-10 13-54-23 120','P203160901_2022-11-10 14-02-59 673'],'20221109-5921pd10')#5
loaddata(['P203160901_2022-11-10 14-38-41 030','P203160901_2022-11-10 14-47-17 096'],'20221109-5959pd10')#1

xtrainlistraw=[]
ytrianlistraw=[]
for i in range(len(Allxlist)):
    xtrainlistraw.extend(Allxlist[i])
    ytrianlistraw.extend(Allylist[i])
cntlist=[i for i in range(len(xtrainlistraw))]
random.shuffle(cntlist)
for i in range(len(cntlist)):
    xtrainlist.append([xtrainlistraw[cntlist[i]]])
    ytrianlist.append([ytrianlistraw[cntlist[i]]])
mlpmodel=MyNet()

print('开始训练')
loss = nn.MSELoss()
x = [i for i in range(len(xtrainlist))]
losslist=[]
acclist = []
for epoch in range(1, num_epochs + 1):
    random.shuffle(x)
    lsum = 0
    for i in range(int(len(x)//batch)) :
        nl,np=torch.tensor([xtrainlist[x[i*batch+j]] for j in range(batch)]),\
                   torch.tensor([ytrianlist[x[i*batch+j]] for j in range(batch)]),
        r = lr / (10 + 0.8 * epoch)
        optimizer = torch.optim.Adam(mlpmodel.parameters(), lr=r)
        y_hat = mlpmodel(nl)
        l = loss(y_hat, np).mean()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        lsum = lsum + l.data.item()

        if i % 100 == 0:
            print("epoch:{},loss:{}".format(epoch, l))
    losslist.append(lsum)
    acc=0
    for i in range(len(Allxlist)):
        ypreloglist = [mlpmodel(torch.tensor([[Allxlist[i][ii]]]))[0][0].data.item() for ii in range(len(Allxlist[i]))]
        yprelist = [10 ** ii for ii in ypreloglist]
        ytruelist = [10 ** ii for ii in Allylist[i]]
        stdlist = [abs(ytruelist[i] - yprelist[i]) / ytruelist[i] for i in range(len(yprelist))]
        std = 0
        for ii in range(len(stdlist)//2,len(stdlist)):
            std = std + stdlist[ii]
        acc = 1 - std / (len(stdlist)//2)+acc
    acc=acc/len(Allxlist)
    acclist.append(acc)
print('训练完成')
net = mlpmodel.to(torch.device('cpu'))

torch.save(net, 'mydata/' + mod + 'net.pkl')
print(losslist)
print(acclist)
for i in range(len(Allxlist)):
    ypreloglist = [mlpmodel(torch.tensor([[Allxlist[i][ii]]]))[0][0].data.item() for ii in range(len(Allxlist[i]))]
    yprelist=[10**ii for ii in ypreloglist]
    ytruelist=[10**ii for ii in Allylist[i]]
    stdlist=[abs(ytruelist[i]-yprelist[i])/ytruelist[i] for i in range(len(yprelist))]
    std=0
    for ii in range(len(stdlist)):
        std=std+stdlist[ii]
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(ytruelist,c='b',label='预测气压  准确度: ' + str(round(std,3))+'%')
    plt.plot(yprelist,c='g')
    plt.legend()
    plt.show()

