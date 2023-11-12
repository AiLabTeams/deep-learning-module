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
mod = "cnnmex20230106"
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
channel1=1
channel2=1
channel3=1
channel4=1
batch=1
lr, num_epochs = 0.03 , 300
channelmut=8


timelen=4
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.convs = nn.Sequential(
           nn.Conv2d(inchanel, channel1*timelen, 3, 2),  # in_channels, out_channels,kernel_size
           nn.Sigmoid(),
           nn.Conv2d(channel1*timelen, channel2*timelen, 2, 2),
           nn.BatchNorm2d(channel2*timelen),
           nn.Sigmoid(),
           nn.Conv2d(channel2*timelen, channel3*timelen, 4, 1),
           nn.BatchNorm2d(channel3*timelen),
           nn.Sigmoid(),
           nn.Conv2d(channel3*timelen, channel3*timelen, 4, 1),
           nn.BatchNorm2d(channel3*timelen),
           nn.Sigmoid(),
           nn.Conv2d(channel3*timelen, channel4*timelen, 3, 1),
           nn.BatchNorm2d(channel4*timelen),
           nn.Sigmoid(),
       )
       self.convd = nn.Sequential(
           nn.Conv2d(inchanel, channel1*timelen, 3, 2),  # in_channels, out_channels,kernel_size
           nn.Sigmoid(),
           nn.Conv2d(channel1*timelen, channel2*timelen, 2, 2),
           nn.BatchNorm2d(channel2*timelen),
           nn.Sigmoid(),
           nn.Conv2d(channel2*timelen, channel3*timelen, 4, 1),
           nn.BatchNorm2d(channel3*timelen),
           nn.Sigmoid(),
           nn.Conv2d(channel3*timelen, channel3*timelen, 4, 1),
           nn.BatchNorm2d(channel3*timelen),
           nn.Sigmoid(),
           nn.Conv2d(channel3*timelen, channel4*timelen, 3, 1),
           nn.BatchNorm2d(channel4*timelen),
           nn.Sigmoid(),
       )
       self.fcs = nn.Sequential(
           nn.Linear(3 * 3 * channel4*timelen, 1),
       )
       self.fcd = nn.Sequential(
           nn.Linear(3 * 3 * channel4*timelen, 1),
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
        PressMove = 0.013
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
cnnmodel = torch.load('mydata/' + mod + 'net.pkl')

#loaddata(['P203160901_2022-10-28 15-32-10 451','P203160901_2022-10-28 15-41-13 123'],'20221027-2427pd10ag10pd10')#c
#loaddata(['P203160901_2022-10-30 19-26-09 817','P203160901_2022-10-30 19-35-08 409'],'20221027-3331pd10ag10pd10')#c
#loaddata(['P203160901_2022-10-27 20-12-30 255','P203160901_2022-10-27 20-21-30 389'],'20221027-4060pd10ag10pd10')#c
#loaddata(['P203160901_2022-10-30 16-55-57 889','P203160901_2022-10-30 17-05-01 953'],'20221027-4587pd10ag10pd10')#c
#loaddata(['P203160901_2022-10-29 16-49-17 698','P203160901_2022-10-29 16-58-20 302'],'20221027-5275pd10ag10pd10')#c
#loaddata(['P203160901_2022-10-28 13-25-43 171','P203160901_2022-10-28 13-34-43 791'],'20221027-6411pd10ag10pd10')#c
#loaddata(['P203160901_2022-10-29 15-50-30 300','P203160901_2022-10-29 15-59-26 939'],'20221027-9056pd10ag10pd10')#c

#loaddata(['P203160901_2022-11-09 20-51-08 739','P203160901_2022-11-09 20-59-46 282'],'20221109-2847pd10')#3
#loaddata(['P203160901_2022-11-10 15-00-01 436','P203160901_2022-11-10 15-08-39 531'],'20221109-3273pd10')#3
#loaddata(['P203160901_2022-11-11 15-38-49 488','P203160901_2022-11-11 15-47-25 551'],'20221109-4218pd10')#3
#loaddata(['P203160901_2022-11-10 17-25-41 128','P203160901_2022-11-10 17-34-17 216'],'20221109-5112pd10')#3
#loaddata(['P203160901_2022-11-09 20-29-12 446','P203160901_2022-11-09 20-37-49 025'],'20221109-6224pd10')#3
#loaddata(['P203160901_2022-11-10 20-46-19 464','P203160901_2022-11-10 20-54-55 053'],'20221109-8826pd10')#3
#loaddata(['P203160901_2022-11-11 14-36-16 187','P203160901_2022-11-11 14-44-52 296'],'20221109-17760pd10')#3

loaddata(['P203160901_2022-11-09 19-46-56 710','P203160901_2022-11-09 19-55-32 268'],'20221109-3794pd10')#1
loaddata(['P203160901_2022-11-10 14-17-28 710','P203160901_2022-11-10 14-26-04 251'],'20221109-4395pd10')#4
for i in range(len(Allxlist)):
    ypreloglist = [cnnmodel(torch.tensor([Allxlist[i][ii][0]]),torch.tensor([Allxlist[i][ii][1]]))[0][0].data.item() for ii in range(len(Allxlist[i]))]
    yprelist=[10**ii for ii in ypreloglist]
    ytruelist=[10**ii for ii in Allylist[i]]
    print(yprelist)
    print(ytruelist)
    #ytruelist = [0 for ii in Allylist[i]]
    #yprelist = [Allxlist[i][ii][1][0]for ii in range(len(Allxlist[i]))]
    stdlist=[abs(ytruelist[i]-yprelist[i]) for i in range(len(yprelist)*6//10,len(yprelist))]
    sumlist = [yprelist[i] for i in range(len(yprelist) * 6 // 10, len(yprelist))]
    std=0
    sum=0
    dx=0
    for ii in range(len(stdlist)):
        std=std+stdlist[ii]
        sum=sum+sumlist[ii]
    sum=sum/len(sumlist)
    std=std/len(stdlist)/sum*100
    for ii in range(len(stdlist)):
        dx=abs(sum-sumlist[ii])+dx
    dx=dx/len(sumlist)/sum*100
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(ytruelist,c='b',label='预测气压  准确度: ' + str(round(std,3))+'%')
    plt.plot(yprelist,c='g',label='预测气压  稳定度: ' + str(round(dx,3))+'%')
    plt.legend()
    plt.show()
