from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import WavePackage
import PressPackage
import math
from scipy.optimize import curve_fit
from scipy.fftpack import fft,ifft
import random
import Gramme
import WavePackage
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torch import nn
mod = "cnns20221129"
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
GASFMatrixListRaw = []
GADFMatrixListRaw = []
GASFMatrixList = []
GADFMatrixList = []
AllGASFMatrixList = []
AllGADFMatrixList = []

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
inchanel=10
channel1=30
channel2=30
channel3=30
channel4=30
batch=1
lr, num_epochs = 0.005 , 100
channelmut=8
refuc=nn.Sigmoid
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.convs = nn.Sequential(
           nn.Conv2d(inchanel, channel1, 3, 2),  # in_channels, out_channels,kernel_size
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
           nn.Conv2d(channel3, channel4, 3, 1),
           nn.BatchNorm2d(channel4),
           nn.Sigmoid(),
       )
       self.convd = nn.Sequential(
           nn.Conv2d(inchanel, channel1, 3, 2),  # in_channels, out_channels,kernel_size
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
           nn.Conv2d(channel3, channel4, 3, 1),
           nn.BatchNorm2d(channel4),
           nn.Sigmoid(),
       )
#       self.covf=nn.Sequential(nn.Conv2d(channel4*2, 1,3,1),)

#    def forward(self, imgs,imgd):
#       features = self.convs(imgs)
#       featured = self.convd(imgd)
#       output= self.covf(torch.cat((features,featured),1))
#       return output

       self.fcs = nn.Sequential(
           nn.Linear(3 * 3 * channel4, 1),
       )
       self.fcd = nn.Sequential(
           nn.Linear(3 * 3 * channel4, 1),
       )

       self.fc = nn.Sequential(
           nn.Linear(3 * 3 * channel4*2, 1)
       )
   def forward(self, imgs,imgd):
       features = self.convs(imgs)
       featured = self.convd(imgd)
       #outputs = self.fcs(features.view(imgs.shape[0], -1))
       #outputd = self.fcd(featured.view(imgs.shape[0], -1))
       #output = self.fc(torch.cat((outputs,outputd),1))
       output = self.fc(torch.cat((features.view(imgs.shape[0], -1), featured.view(imgs.shape[0], -1)), 1))
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
        PressMove = 0.0115
        temp = 0
        pmin = min(PressAmplitudeList)
        PressAmplitudeList=[i-pmin for i in PressAmplitudeList]
        for i in range(len(PressAmplitudeList)):
            temp = temp + PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] > 10:
                temp = temp - PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] < -10:
                temp = 0
            NewPressAmplitudeList.append(max(PressAmplitudeList[i] - temp, 2))
            #NewPressAmplitudeList.append(min(PressAmplitudeList[i], PressAmplitudeList[len(PressAmplitudeList)-1]*1.01))
        PressAmplitudelogList = [math.log(max(2, i), 10) for i in NewPressAmplitudeList]
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
def GASFFunc():
    global GASFMatrixListRaw
    global GASFMatrixList
    for i in LightWaveAmplitudeLists:
        listtemp = []
        for ii in range(0, len(i), 15):
            if(LightWaveLengthList[ii]>600 and LightWaveLengthList[ii]<900):
                            listtemp.append(i[ii])
        temp=Gramme.GASFExchange(listtemp,flag=0)
        GASFMatrixListRaw.append(temp)
    GASFMatrixList=GASFMatrixListRaw.copy()
    print('GASF 成功')
def GADFFunc():
    global GADFMatrixListRaw
    global GADFMatrixList
    for i in LightWaveAmplitudeLists:
        listtemp = []
        for ii in range(0, len(i), 15):
            if(LightWaveLengthList[ii]>600 and LightWaveLengthList[ii]<900):
                            listtemp.append(i[ii])
        temp=Gramme.GADFExchange(listtemp,flag=0)
        GADFMatrixListRaw.append(temp)
    GADFMatrixList=GADFMatrixListRaw.copy()
    print('GADF 成功')


def Dataget():
    global Allxlist
    global Allylist
    xlist=[]
    ylist=[]
    if abs(PressTimeToFix - LightTimeToFix) > 36000:
        temp = PressTimeToFix - LightTimeToFix + 12 * 60 * 60
    else:
        temp = PressTimeToFix - LightTimeToFix

    for pt in range(timedis*8//10+50,timedis+50):
        for lt in range(inchanel+5, len(LightWaveTimeSecondList)):
            if LightWaveTimeSecondList[lt] >= PressTimeList[pt] + temp:
                templist = [[], []]
                # templist.extend(fftabslistlist[lt].copy())
                # templist.extend(fftanglelist[lt].copy())
                # templist.extend(LightWaveAmplitudeLists[lt][::15])
                templist[0].extend(GASFMatrixList[lt-inchanel:lt])
                templist[1].extend(GADFMatrixList[lt-inchanel:lt])
                xlist.append(templist.copy())
                ylist.append(PressAmplitudelogList[pt])
                break
    for pt in range(len(PressTimeList)-timedis, len(PressTimeList)):
        for lt in range(inchanel+5, len(LightWaveTimeSecondList)):
            if LightWaveTimeSecondList[lt] >= PressTimeList[pt] + temp:
                templist = [[], []]
                # templist.extend(fftabslistlist[lt].copy())
                # templist.extend(fftanglelist[lt].copy())
                # templist.extend(LightWaveAmplitudeLists[lt][::15])
                templist[0].extend(GASFMatrixList[lt-inchanel:lt])
                templist[1].extend(GADFMatrixList[lt-inchanel:lt])
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
    global GASFMatrixListRaw
    global GADFMatrixListRaw
    global GASFMatrixList
    global GADFMatrixList
    global AllGASFMatrixList
    global AllGADFMatrixList
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
    AllGASFMatrixList.append(GASFMatrixList.copy())
    AllGADFMatrixList.append(GADFMatrixList.copy())
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
    GASFMatrixListRaw = []
    GADFMatrixListRaw = []
    GASFMatrixList = []
    GADFMatrixList = []
    lightfilecount = 0
    print('清除缓存')
def loaddata(lpaths,ppath):
    for i in range(len(lpaths)):
        Import_light_file('mydata/'+lpaths[i]+'.txt')
    Import_press_file('mydata/'+ppath+'.csv')
    zero()
    fftfuc()
    SGDfuc()
    GADFFunc()
    GASFFunc()
    Dataget()
    next()






#loaddata(['P203160901_2022-11-11 22-03-22 653','P203160901_2022-11-11 22-12-04 242'],'20221109-181pd10')
##loaddata(['P203160901_2022-11-11 20-43-15 527','P203160901_2022-11-11 20-51-53 664'],'20221109-189pd10')
#loaddata(['P203160901_2022-11-11 21-43-43 862','P203160901_2022-11-11 21-52-24 512'],'20221109-191pd10')
#loaddata(['P203160901_2022-11-11 22-21-29 345','P203160901_2022-11-11 22-30-12 686'],'20221109-198d10')
#loaddata(['P203160901_2022-11-11 21-23-51 234','P203160901_2022-11-11 21-32-30 778'],'20221109-204pd10')
##loaddata(['P203160901_2022-11-11 21-04-47 098','P203160901_2022-11-11 21-13-22 650'],'20221109-208pd10')
#loaddata(['P203160901_2022-11-11 19-59-46 681','P203160901_2022-11-11 20-08-21 795'],'20221109-219pd10')
#loaddata(['P203160901_2022-11-11 17-12-25 492','P203160901_2022-11-11 17-21-02 133'],'20221109-254pd10')
#loaddata(['P203160901_2022-11-11 20-20-23 980','P203160901_2022-11-11 20-28-59 554'],'20221109-257pd10')
##loaddata(['P203160901_2022-11-09 14-16-00 015','P203160901_2022-11-09 14-24-36 083'],'20221109-291pd10')
#loaddata(['P203160901_2022-11-11 19-39-45 346','P203160901_2022-11-11 19-48-20 922'],'20221109-352pd10')
#loaddata(['P203160901_2022-11-09 15-35-56 657','P203160901_2022-11-09 15-44-33 264'],'20221109-447pd10')
#loaddata(['P203160901_2022-11-09 13-24-51 311','P203160901_2022-11-09 13-33-26 918'],'20221109-477pd10')
##loaddata(['P203160901_2022-11-11 16-46-22 660','P203160901_2022-11-11 16-54-59 269'],'20221109-781pd10')
#loaddata(['P203160901_2022-11-11 18-33-56 947','P203160901_2022-11-11 18-42-33 034'],'20221109-782pd10')
#loaddata(['P203160901_2022-11-09 13-47-58 156','P203160901_2022-11-09 13-56-32 681'],'20221109-804pd10')
#loaddata(['P203160901_2022-11-09 15-03-44 326','P203160901_2022-11-09 15-12-21 911'],'20221109-855pd10')
##loaddata(['P203160901_2022-11-09 14-40-16 824','P203160901_2022-11-09 14-48-52 902'],'20221109-1023pd10')
#loaddata(['P203160901_2022-11-09 17-02-50 731','P203160901_2022-11-09 17-11-27 287','P203160901_2022-11-09 17-20-04 278'],'20221109-1356pd10')
#loaddata(['P203160901_2022-11-09 19-25-50 364','P203160901_2022-11-09 19-34-27 023'],'20221109-1534pd10')
#loaddata(['P203160901_2022-11-09 15-58-27 878','P203160901_2022-11-09 16-07-04 413'],'20221109-1558pd10')
#loaddata(['P203160901_2022-11-10 12-56-45 199','P203160901_2022-11-10 13-05-21 777'],'20221109-1971pd10')
#
#loaddata(['P203160901_2022-11-10 15-21-02 904','P203160901_2022-11-10 15-29-38 926'],'20221109-2432pd10')#1
loaddata(['P203160901_2022-11-10 15-43-34 744','P203160901_2022-11-10 15-52-10 297'],'20221109-2661pd10')#2
loaddata(['P203160901_2022-11-09 20-51-08 739','P203160901_2022-11-09 20-59-46 282'],'20221109-2847pd10')#3
loaddata(['P203160901_2022-11-09 16-40-45 813','P203160901_2022-11-09 16-49-22 527'],'20221109-3047pd10')#4
loaddata(['P203160901_2022-11-10 21-49-48 028','P203160901_2022-11-10 21-58-25 088'],'20221109-3152pd10')#5
#loaddata(['P203160901_2022-11-09 16-20-32 054','P203160901_2022-11-09 16-29-08 129'],'20221109-3201pd10')#1
loaddata(['P203160901_2022-11-11 16-01-06 530','P203160901_2022-11-11 16-09-42 663'],'20221109-3226pd10')#2
loaddata(['P203160901_2022-11-10 15-00-01 436','P203160901_2022-11-10 15-08-39 531'],'20221109-3273pd10')#3
loaddata(['P203160901_2022-11-10 13-29-35 324','P203160901_2022-11-10 13-38-11 975'],'20221109-3493pd10')#4
loaddata(['P203160901_2022-11-10 16-03-56 353','P203160901_2022-11-10 16-12-32 459'],'20221109-3685pd10')#5
#loaddata(['P203160901_2022-11-09 19-46-56 710','P203160901_2022-11-09 19-55-32 268'],'20221109-3794pd10')#1
loaddata(['P203160901_2022-11-11 14-14-44 805','P203160901_2022-11-11 14-23-20 424'],'20221109-3833pd10')#2
loaddata(['P203160901_2022-11-11 15-38-49 488','P203160901_2022-11-11 15-47-25 551'],'20221109-4218pd10')#3
loaddata(['P203160901_2022-11-10 14-17-28 710','P203160901_2022-11-10 14-26-04 251'],'20221109-4395pd10')#4
loaddata(['P203160901_2022-11-09 21-12-55 253','P203160901_2022-11-09 21-21-31 775'],'20221109-4577pd10')#5
#loaddata(['P203160901_2022-11-10 22-29-18 171','P203160901_2022-11-10 22-37-55 972'],'20221109-4595pd10')#1
loaddata(['P203160901_2022-11-10 17-48-11 735','P203160901_2022-11-10 17-56-47 297'],'20221109-4883pd10')#2
loaddata(['P203160901_2022-11-10 17-25-41 128','P203160901_2022-11-10 17-34-17 216'],'20221109-5112pd10')#3
loaddata(['P203160901_2022-11-10 17-04-44 910','P203160901_2022-11-10 17-13-20 460'],'20221109-5687pd10')#4
loaddata(['P203160901_2022-11-10 13-54-23 120','P203160901_2022-11-10 14-02-59 673'],'20221109-5921pd10')#5
#loaddata(['P203160901_2022-11-10 14-38-41 030','P203160901_2022-11-10 14-47-17 096'],'20221109-5959pd10')#1
loaddata(['P203160901_2022-11-10 12-36-40 814','P203160901_2022-11-10 12-45-17 441'],'20221109-6009pd10')#2
loaddata(['P203160901_2022-11-09 20-29-12 446','P203160901_2022-11-09 20-37-49 025'],'20221109-6224pd10')#3
loaddata(['P203160901_2022-11-11 17-44-31 868','P203160901_2022-11-11 17-53-07 401'],'20221109-6866pd10')#4
loaddata(['P203160901_2022-11-11 16-24-30 680','P203160901_2022-11-11 16-33-07 294'],'20221109-7235pd10')#5
#loaddata(['P203160901_2022-11-10 16-23-54 181','P203160901_2022-11-10 16-32-30 284'],'20221109-7387pd10')#1
loaddata(['P203160901_2022-11-10 16-45-11 576','P203160901_2022-11-10 16-53-47 680'],'20221109-8618pd10')#2
loaddata(['P203160901_2022-11-10 20-46-19 464','P203160901_2022-11-10 20-54-55 053'],'20221109-8826pd10')#3
loaddata(['P203160901_2022-11-10 22-11-14 512','P203160901_2022-11-10 22-19-50 563'],'20221109-11406pd10')#4
loaddata(['P203160901_2022-11-10 21-07-58 237','P203160901_2022-11-10 21-16-33 287'],'20221109-14708pd10')#5
#loaddata(['P203160901_2022-11-11 15-17-51 613','P203160901_2022-11-11 15-26-27 176'],'20221109-15780pd10')#1
loaddata(['P203160901_2022-11-10 21-30-24 668','P203160901_2022-11-10 21-38-59 751'],'20221109-15875pd10')#2
loaddata(['P203160901_2022-11-11 14-36-16 187','P203160901_2022-11-11 14-44-52 296'],'20221109-17760pd10')#3
loaddata(['P203160901_2022-11-11 14-57-36 651'],'20221109-19960pd10')#4





xtrainlistraw=[]
ytrianlistraw=[]
for i in range(len(Allxlist)):
    xtrainlistraw.extend(Allxlist[i])
    ytrianlistraw.extend(Allylist[i])
cntlist=[i for i in range(len(xtrainlistraw))]
random.shuffle(cntlist)
for i in range(len(cntlist)):
    xtrainlist.append(xtrainlistraw[cntlist[i]])
    ytrianlist.append([ytrianlistraw[cntlist[i]]])
cnnmodel=MyNet()

print('开始训练')
loss = nn.MSELoss()
x = [i for i in range(len(xtrainlist))]
lrcnt=0
losslist=[]
for epoch in range(1, num_epochs + 1):
    random.shuffle(x)
    lsum=0
    for i in range(int(len(x)//batch)) :
        lrcnt = lrcnt + 1
        if lrcnt > 100:
            lrcnt = 1
        nl1,nl2,np=torch.tensor([xtrainlist[x[i*batch+j]][0] for j in range(batch)]),torch.tensor([xtrainlist[x[i*batch+j]][1] for j in range(batch)]),\
                   torch.tensor([[[ytrianlist[x[i*batch+j]]]] for j in range(batch)]),
        #r = lr / (10 + lrcnt)
        #r = lr / (10 + 0.6*epoch)
        r=lr/10
        optimizer = torch.optim.Adam(cnnmodel.parameters(), lr=r)
        y_hat= cnnmodel(nl1,nl2)
        l = loss(y_hat, np).mean()
        optimizer.zero_grad()
        l.backward()
        lsum=lsum+l.data.item()

        optimizer.step()
        if i % 100 == 0:
            print("epoch:{},loss:{}".format(epoch, l))
    losslist.append(lsum)
print('训练完成')
net = cnnmodel.to(torch.device('cpu'))
torch.save(net, 'mydata/' + mod + 'net.pkl')



##loaddata(['P203160901_2022-09-15 16-50-49 287'],'20220915-467pd20')
##loaddata(['P203160901_2022-09-15 17-08-27 954'],'20220915-189pd20')
##loaddata(['P203160901_2022-09-15 19-54-09 013'],'20220915-899pd20')
##loaddata(['P203160901_2022-09-16 11-20-53 697'],'20220915-1699pd20')
#
#loaddata(['P203160901_2022-09-16 13-13-03 508'],'20220915-4056pd20')
#loaddata(['P203160901_2022-09-16 14-27-47 426'],'20220915-3100pd20')
#loaddata(['P203160901_2022-09-16 12-15-13 206'],'20220915-2139pd20')
#loaddata(['P203160901_2022-09-15 20-30-25 903'],'20220915-6289pd20')
print(losslist)
plt.plot(losslist)
plt.show()
for i in range(len(Allxlist)):
    if i==len(Allxlist)-1-3:
        print(" test ")
    ypreloglist = [cnnmodel(torch.tensor([Allxlist[i][ii][0]]),torch.tensor([Allxlist[i][ii][1]]))[0][0].data.item() for ii in range(len(Allxlist[i]))]
    yprelist=[10**ii for ii in ypreloglist]
    ytruelist=[10**ii for ii in Allylist[i]]
    stdlist=[abs(ytruelist[i]-yprelist[i]) for i in range(len(yprelist))]
    std=0
    for ii in range(5,len(stdlist)):
        std=std+stdlist[ii]
    std=std/(len(stdlist)-5)
    stlen=len(yprelist)//2-1
    avelist = [yprelist[i] for i in range(len(yprelist) - stlen, len(yprelist))]
    ave = 0
    for ii in range(len(avelist)):
        ave = ave + avelist[ii]
    ave = ave /len(avelist)
    stlist = [abs(avelist[i]-ave) for i in range( len(avelist))]
    st = 0
    for ii in range(len(stlist)):
        st = st + stlist[ii]
    st = st / len(stlist)
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(ytruelist,c='b',label='原始气压  准确度: ' + str(round(std,3)))
    plt.plot(yprelist,c='g',label='预测气压  稳定度: ' + str(round(st,3)))
    plt.legend()
    plt.show()
